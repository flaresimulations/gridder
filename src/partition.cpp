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
  for (int cid = 0; cid < sim->nr_cells; cid++) {
    // Get the cell
    std::shared_ptr<Cell> cell = sim->cells[cid];

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
  const int size = metadata->size;

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
        std::shared_ptr<Cell> ci = sim->cells[cid];

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
              std::shared_ptr<Cell> cj = sim->cells[cjd];

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
 * the appropriate ranks (either sending to multiple ranks or receiving from a
 * rank). Particle data includes masses and positions.
 *
 * @param sim Pointer to the Simulation object containing cells and particles.
 */
void exchangeProxyCells(Simulation *sim) {
  // Get the metadata and MPI info
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;
  const int size = metadata->size;

  // Loop over all cells in the simulation
  for (int cid = 0; cid < sim->nr_cells; cid++) {
    // Get the cell
    std::shared_ptr<Cell> cell = sim->cells[cid];

    // Skip cells that have no communication (no send or receive ranks)
    if (cell->send_ranks.empty() && cell->recv_rank == -1) {
      continue;
    }

    // Determine how many particles will be exchanged
    int npart_exchange = cell->part_count;

    // Create arrays to hold particle data (masses and positions)
    std::vector<double> send_masses(npart_exchange);
    std::vector<double> send_poss(3 * npart_exchange);

    // Receiving particles
    if (cell->recv_rank != -1) {
      // Receive masses and positions from the specified rank
      MPI_Recv(send_masses.data(), npart_exchange, MPI_DOUBLE, cell->recv_rank,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(send_poss.data(), 3 * npart_exchange, MPI_DOUBLE,
               cell->recv_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Loop over received particles and attach them to the cell
      for (int p = 0; p < npart_exchange; p++) {
        // Extract the mass and position of each particle
        const double mass = send_masses[p];
        const double pos[3] = {send_poss[p * 3], send_poss[p * 3 + 1],
                               send_poss[p * 3 + 2]};

        // Create a new particle and handle any exceptions
        try {
          std::shared_ptr<Particle> part =
              std::make_shared<Particle>(pos, mass);

          // Update the cell's total mass
          cell->mass += mass;

          // Add the particle to the cell's particle list
          cell->particles.push_back(part);
        } catch (const std::exception &e) {
          error("Failed to create particle: %s", e.what());
        }
      }
    }
    // Sending particles
    else {
      // Populate send buffers with particle data (masses and positions)
      for (int p = 0; p < npart_exchange; p++) {
        send_masses[p] = cell->particles[p]->mass;
        send_poss[p * 3] = cell->particles[p]->pos[0];
        send_poss[p * 3 + 1] = cell->particles[p]->pos[1];
        send_poss[p * 3 + 2] = cell->particles[p]->pos[2];
      }

      // Send particle data to all ranks in send_ranks
      for (int send_rank : cell->send_ranks) {
        MPI_Send(send_masses.data(), npart_exchange, MPI_DOUBLE, send_rank, 0,
                 MPI_COMM_WORLD);
        MPI_Send(send_poss.data(), 3 * npart_exchange, MPI_DOUBLE, send_rank, 1,
                 MPI_COMM_WORLD);
      }
    }
  }
}
#endif
