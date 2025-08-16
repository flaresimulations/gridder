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
/**
 * @brief Pack particles for all proxy cells destined for each rank
 *
 * @param sim Simulation object
 * @param rank_data Map from destination rank to packed particle data
 */
void packProxyComms(Simulation *sim, std::map<int, std::vector<double>>& rank_masses,
                    std::map<int, std::vector<double>>& rank_positions,
                    std::map<int, std::vector<int>>& rank_cell_ids,
                    std::map<int, std::vector<int>>& rank_cell_counts) {
  
  // Get the current rank
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;
  
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell* cell = &sim->cells[cid];
    
    // Only send cells that we own and have send ranks
    if (cell->rank != rank || cell->send_ranks.empty()) continue;
    
    int npart = cell->part_count;
    
    // For each rank that needs this cell as a proxy
    for (int dest_rank : cell->send_ranks) {
      // Add cell metadata
      rank_cell_ids[dest_rank].push_back(cid);
      rank_cell_counts[dest_rank].push_back(npart);
      
      // Pack particle data for this cell
      for (int p = 0; p < npart; p++) {
        rank_masses[dest_rank].push_back(cell->particles[p]->mass);
        rank_positions[dest_rank].push_back(cell->particles[p]->pos[0]);
        rank_positions[dest_rank].push_back(cell->particles[p]->pos[1]);
        rank_positions[dest_rank].push_back(cell->particles[p]->pos[2]);
      }
    }
  }
}

/**
 * @brief Unpack received particles and assign to proxy cells
 *
 * @param sim Simulation object
 * @param masses Received masses data
 * @param positions Received positions data  
 * @param cell_ids Received cell IDs
 * @param cell_counts Received particle counts per cell
 */
void unpackProxyComms(Simulation *sim, const std::vector<double>& masses,
                      const std::vector<double>& positions,
                      const std::vector<int>& cell_ids,
                      const std::vector<int>& cell_counts) {
  
  int mass_offset = 0;
  int pos_offset = 0;
  
  for (size_t i = 0; i < cell_ids.size(); i++) {
    int cid = cell_ids[i];
    int npart = cell_counts[i];
    Cell* cell = &sim->cells[cid];
    
    // Create particles for this proxy cell
    for (int p = 0; p < npart; p++) {
      const double mass = masses[mass_offset + p];
      const double pos[3] = {positions[pos_offset + p * 3],
                             positions[pos_offset + p * 3 + 1],
                             positions[pos_offset + p * 3 + 2]};
      
      // Create the particle using raw pointer allocation
      Particle* part = new Particle(pos, mass);
      
      // Update the cell's total mass and attach particle
      cell->mass += mass;
      cell->particles.push_back(part);
    }
    
    // Update particle count and offsets
    cell->part_count = npart;
    mass_offset += npart;
    pos_offset += npart * 3;
  }
}

void exchangeProxyCells(Simulation *sim) {
  // Get the metadata and MPI info
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;
  const int size = metadata->size;

  message("Rank %d: Starting proxy cell exchange", rank);

  // Pack data for each destination rank
  std::map<int, std::vector<double>> rank_masses, rank_positions;
  std::map<int, std::vector<int>> rank_cell_ids, rank_cell_counts;
  
  packProxyComms(sim, rank_masses, rank_positions, rank_cell_ids, rank_cell_counts);
  
  // Report packing statistics
  int total_send_cells = 0, total_send_particles = 0;
  for (const auto& pair : rank_cell_ids) {
    total_send_cells += pair.second.size();
    total_send_particles += rank_masses[pair.first].size();
  }
  message("Rank %d: Packed %d cells with %d particles for sending", rank, total_send_cells, total_send_particles);

  // Send packed data to each rank (non-blocking)
  std::vector<MPI_Request> requests;
  int send_count = 0;
  
  for (int dest_rank = 0; dest_rank < size; dest_rank++) {
    if (dest_rank == rank || rank_masses[dest_rank].empty()) continue;
    
    // Send metadata first
    int num_cells = rank_cell_ids[dest_rank].size();
    int num_particles = rank_masses[dest_rank].size();
    message("Rank %d: Sending %d cells (%d particles) to rank %d", rank, num_cells, num_particles, dest_rank);
    
    MPI_Request req1, req2, req3, req4;
    
    MPI_Isend(&num_cells, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, &req1);
    MPI_Isend(rank_cell_ids[dest_rank].data(), num_cells, MPI_INT, dest_rank, 1, MPI_COMM_WORLD, &req2);
    MPI_Isend(rank_cell_counts[dest_rank].data(), num_cells, MPI_INT, dest_rank, 2, MPI_COMM_WORLD, &req3);
    
    // Send particle data
    MPI_Isend(rank_masses[dest_rank].data(), rank_masses[dest_rank].size(), MPI_DOUBLE, dest_rank, MPI_TAG_MASS, MPI_COMM_WORLD, &req4);
    
    requests.insert(requests.end(), {req1, req2, req3, req4});
    
    if (!rank_positions[dest_rank].empty()) {
      MPI_Request req5;
      MPI_Isend(rank_positions[dest_rank].data(), rank_positions[dest_rank].size(), MPI_DOUBLE, dest_rank, MPI_TAG_POSITION, MPI_COMM_WORLD, &req5);
      requests.push_back(req5);
    }
    send_count++;
  }
  
  message("Rank %d: Posted sends to %d ranks", rank, send_count);

  // Receive data from each rank
  int recv_count = 0;
  for (int src_rank = 0; src_rank < size; src_rank++) {
    if (src_rank == rank) continue;
    
    // Check if we expect data from this rank
    bool expect_data = false;
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      if (sim->cells[cid].recv_rank == src_rank) {
        expect_data = true;
        break;
      }
    }
    
    if (!expect_data) continue;
    
    message("Rank %d: Expecting proxy data from rank %d", rank, src_rank);
    
    // Receive metadata
    int num_cells;
    MPI_Recv(&num_cells, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    std::vector<int> cell_ids(num_cells), cell_counts(num_cells);
    MPI_Recv(cell_ids.data(), num_cells, MPI_INT, src_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(cell_counts.data(), num_cells, MPI_INT, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Calculate total particles
    int total_particles = 0;
    for (int count : cell_counts) total_particles += count;
    
    message("Rank %d: Receiving %d cells (%d particles) from rank %d", rank, num_cells, total_particles, src_rank);
    
    // Receive particle data
    std::vector<double> masses(total_particles), positions(total_particles * 3);
    MPI_Recv(masses.data(), total_particles, MPI_DOUBLE, src_rank, MPI_TAG_MASS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(positions.data(), total_particles * 3, MPI_DOUBLE, src_rank, MPI_TAG_POSITION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Unpack received data
    unpackProxyComms(sim, masses, positions, cell_ids, cell_counts);
    recv_count++;
  }
  
  message("Rank %d: Received data from %d ranks", rank, recv_count);
  
  // Wait for all sends to complete
  if (!requests.empty()) {
    message("Rank %d: Waiting for %zu send operations to complete", rank, requests.size());
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
  
  message("Rank %d: Proxy cell exchange completed", rank);
}
#endif

