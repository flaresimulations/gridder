// Standard includes
#include <algorithm>
#include <cmath>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "metadata.hpp"
#include "particle.hpp"
#include "partition.hpp"
#include "simulation.hpp"

#ifdef WITH_MPI
// MPI communication tags
static constexpr int MPI_TAG_MASS = 0;
static constexpr int MPI_TAG_POSITION = 1;
/**
 * @brief Function to partition the cells over the MPI ranks.
 *
 * This function will partition the cells over the MPI ranks such that each rank
 * has an equal number of particles.
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
void partitionCells(Simulation *sim, Grid *grid) {
  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get MPI rank and size
  const int rank = metadata->rank;
  const int size = metadata->size;

  // Intialise an array to hold the number of particles on each rank
  std::vector<int> rank_part_counts(size, 0);

  // How many particles would we expect per rank for a perfect partition?
  const int particles_per_rank = sim->nr_dark_matter / size;

  // Loop over cells and assign them to ranks such that the first
  // particles_per_rank are on rank 0, the next particles_per_rank on rank 1,
  // etc.
  int select = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    // Get the cell
    Cell* cell = &sim->cells[cid];

    // Get the number of particles in the cell
    int part_count = cell->part_count;

    // Assign this rank to the cell
    cell->rank = select;

    // If were assigning to this rank then increment the particle and cell
    // counts
    if (select == rank) {
      metadata->nr_local_cells++;
      metadata->nr_local_particles += part_count;
    }

    // If this is our first local cell set the first local particle index
    if (metadata->first_local_part_ind == -1 && select == rank) {
      metadata->first_local_part_ind = sim->cell_part_starts[cid];
    }

    // Increment the particle count for this rank
    rank_part_counts[select] += part_count;

    // If we have assigned enough particles to this rank, move to the next
    if (rank_part_counts[select] >= particles_per_rank) {
      select++;
    }
  }

  // Report the number of particles on each rank
  message("Rank %d has %d local particles", rank, rank_part_counts[rank]);
  message("Rank %d has %d local cells", rank, metadata->nr_local_cells);
}
#endif

#ifdef WITH_MPI
/**
 * @brief Define the proxy cells we need at the boundaries of the partitions.
 *
 * A proxy is a cell which is within the kernel radius of a cell on this rank.
 * We need to know which cells are within the kernel radius of the boundary
 * so we can communicate the particles that are within the kernel radius of the
 * boundary cells.
 *
 * This function will flag where we need to send our local cells to and where we
 * need to receive cells from.
 *
 * @param sim The simulation object.
 */
void flagProxyCells(Simulation *sim, Grid *grid) {

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get MPI rank and size
  const int rank = metadata->rank;

  // How many cells do we need to walk out from the boundary of the partition
  // to get all the cells that are within the kernel radius of the boundary
  // cells?
  const double max_kernel_radius = grid->max_kernel_radius;
  const double cell_size = sim->width[0];
  const int delta = std::ceil(max_kernel_radius / cell_size) + 1;

  // Loop over cells and find all local
  for (int i = 0; i < sim->cdim[0]; i++) {
    for (int j = 0; j < sim->cdim[1]; j++) {
      for (int k = 0; k < sim->cdim[2]; k++) {
        // Get the cell index
        int cid = i * sim->cdim[1] * sim->cdim[2] + j * sim->cdim[2] + k;

        // Get the cell
        Cell* ci = &sim->cells[cid];

        // Skip useless cells
        if (!ci->is_useful) {
          continue;
        }

        // Loop over the cells in the delta region around this cell
        for (int ii = i - delta; ii <= i + delta; ii++) {
          for (int jj = j - delta; jj <= j + delta; jj++) {
            for (int kk = k - delta; kk <= k + delta; kk++) {
              // Wrap at the boundaries to account for periodicity
              int iii = (ii + sim->cdim[0]) % sim->cdim[0];
              int jjj = (jj + sim->cdim[1]) % sim->cdim[1];
              int kkk = (kk + sim->cdim[2]) % sim->cdim[2];

              // Get the cj possible proxy
              const int cjd =
                  iii * sim->cdim[1] * sim->cdim[2] + jjj * sim->cdim[2] + kkk;
              Cell* cj = &sim->cells[cjd];

              // Ensure no double counts
              if (cjd < cid) {
                continue;
              }

              // Skip useless cells
              if (!cj->is_useful) {
                continue;
              }

              // If ci and cj are both on the same rank there's nothing to do
              if (ci->rank == cj->rank) {
                continue;
              }

              // If both ci and cj are foreign we also don't care
              if (ci->rank != rank && cj->rank != rank) {
                continue;
              }

              // Handle the two cases (either sending ci -> cj, or cj -> ci)
              if (ci->rank == rank) {
                ci->send_ranks.push_back(cj->rank);
                cj->recv_rank = ci->rank;
              } else {
                cj->send_ranks.push_back(ci->rank);
                ci->recv_rank = cj->rank;
              }
            }
          }
        }
      }
    }
  }
}
#endif

#ifdef WITH_MPI
/**
 * @brief Exchange the proxy cells between the MPI ranks.
 *
 * This function loops over all cells and exchanges particle data with
 * the appropriate ranks. Local cells send their particle data to ranks that
 * need them as proxies, and proxy cells receive particle data from the rank
 * that owns them.
 *
 * @param sim Pointer to the Simulation object containing cells and particles.
 */
void exchangeProxyCells(Simulation *sim) {
  // Get the metadata and MPI info
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;

  // Debug: Count and report communication expectations
  int send_count = 0, recv_count = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell* cell = &sim->cells[cid];
    if (!cell->send_ranks.empty()) send_count++;
    if (cell->recv_rank != -1) recv_count++;
  }
  message("Rank %d: expecting to send %d cells, receive %d cells", rank, send_count, recv_count);

  // First pass: Do all the sends (non-blocking to avoid deadlock)
  std::vector<MPI_Request> requests;
  std::vector<std::vector<double>> send_buffers_mass, send_buffers_pos;
  
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell* cell = &sim->cells[cid];
    
    // Sending particles (this is a local cell that others need as proxy)
    if (!cell->send_ranks.empty()) {
      int npart_exchange = cell->part_count;
      
      // Create buffers for this cell's data
      send_buffers_mass.emplace_back(npart_exchange);
      send_buffers_pos.emplace_back(3 * npart_exchange);
      
      auto& send_masses = send_buffers_mass.back();
      auto& send_poss = send_buffers_pos.back();

      // Populate send buffers with particle data from local cell
      for (int p = 0; p < npart_exchange; p++) {
        send_masses[p] = cell->particles[p]->mass;
        send_poss[p * 3] = cell->particles[p]->pos[0];
        send_poss[p * 3 + 1] = cell->particles[p]->pos[1];
        send_poss[p * 3 + 2] = cell->particles[p]->pos[2];
      }

      // Send particle data to all ranks that need this cell as a proxy
      for (int send_rank : cell->send_ranks) {
        MPI_Request req_mass, req_pos;
        MPI_Isend(send_masses.data(), npart_exchange, MPI_DOUBLE, send_rank, 
                  MPI_TAG_MASS, MPI_COMM_WORLD, &req_mass);
        MPI_Isend(send_poss.data(), 3 * npart_exchange, MPI_DOUBLE, send_rank, 
                  MPI_TAG_POSITION, MPI_COMM_WORLD, &req_pos);
        requests.push_back(req_mass);
        requests.push_back(req_pos);
      }
    }
  }

  // Second pass: Do all the receives
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell* cell = &sim->cells[cid];
    
    // Receiving particles (this is a proxy cell)
    if (cell->recv_rank != -1) {
      int npart_exchange = cell->part_count;
      
      // Create arrays to receive particle data (masses and positions)
      std::vector<double> recv_masses(npart_exchange);
      std::vector<double> recv_poss(3 * npart_exchange);

      // Receive masses and positions from the owning rank
      MPI_Recv(recv_masses.data(), npart_exchange, MPI_DOUBLE, cell->recv_rank,
               MPI_TAG_MASS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(recv_poss.data(), 3 * npart_exchange, MPI_DOUBLE,
               cell->recv_rank, MPI_TAG_POSITION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Create particles for this proxy cell
      for (int p = 0; p < npart_exchange; p++) {
        // Extract the mass and position of each particle
        const double mass = recv_masses[p];
        const double pos[3] = {recv_poss[p * 3], recv_poss[p * 3 + 1],
                               recv_poss[p * 3 + 2]};

        // Create the particle using raw pointer allocation
        Particle* part = new Particle(pos, mass);

        // Update the cell's total mass
        cell->mass += mass;

        // Attach the particle to the cell
        cell->particles.push_back(part);
      }
    }
  }
  
  // Wait for all sends to complete
  if (!requests.empty()) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
}
#endif

