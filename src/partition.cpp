// Standard includes
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

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
 */
void partitionCells(Simulation *sim) {
  tic();

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get MPI rank and size
  const int rank = metadata->rank;
  const int size = metadata->size;

  // If we only have one rank, nothing to do
  if (size <= 1) {
    message("Only one MPI rank, no partitioning needed");
    metadata->nr_local_cells = sim->nr_cells;
    metadata->nr_local_particles = sim->nr_dark_matter;
    metadata->first_local_part_ind = 0;
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      sim->cells[cid].rank = 0; // All cells are on rank 0
    }
    toc("Partitioning cells");
    return;
  }

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
    Cell *cell = &sim->cells[cid];

    // Get the number of particles in the cell
    int part_count = cell->part_count;

    // Assign this rank to the cell
    cell->rank = std::min(select, size - 1);

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

#ifdef DEBUGGING_CHECKS
  // Ensure everyone agrees on the cell locations
  std::vector<int> cell_ranks(sim->nr_cells, -1);
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];
    if (cell->rank < 0 || cell->rank >= size) {
      error("Cell %zu has invalid rank %d", cid, cell->rank);
    }
    cell_ranks[cid] = cell->rank;
  }

  // Ensure all ranks have the same cell ranks
  MPI_Allreduce(MPI_IN_PLACE, cell_ranks.data(), sim->nr_cells, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cell_ranks[cid] != sim->cells[cid].rank) {
      error("Cell %zu has rank %d but expected %d", cid, sim->cells[cid].rank,
            cell_ranks[cid]);
    }
  }
#endif

  // Barrier for clarity of output
  MPI_Barrier(MPI_COMM_WORLD);

  toc("Partitioning cells");
}
#endif

/**
 * @brief Prepare particle chunks for efficient reading
 *
 * This function identifies contiguous ranges of useful cells (cells that
 * contain or are neighbors of grid points) and groups them into chunks for
 * efficient HDF5 reading. Since cells are ijk-ordered in the file, contiguous
 * cell IDs correspond to contiguous particle ranges.
 *
 * Works in both serial and MPI modes.
 *
 * @param sim The simulation object
 * @return Vector of ParticleChunk structs describing contiguous ranges
 */
std::vector<ParticleChunk> prepareToReadParts(Simulation *sim) {
  tic();

  std::vector<Cell> &cells = sim->cells;

  // Build sorted list of useful cell IDs
  std::vector<size_t> useful_cells;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].is_useful) {
      useful_cells.push_back(cid);
    }
  }

  if (useful_cells.empty()) {
    message("No useful cells found - nothing to read");
    toc("Preparing to read particles");
    return {};
  }

  message("Found %zu useful cells out of %zu total cells",
          useful_cells.size(), sim->nr_cells);

  // Build chunks where consecutive cells have contiguous particles
  std::vector<ParticleChunk> chunks;
  ParticleChunk current_chunk;

  current_chunk.start_cell_id = useful_cells[0];
  current_chunk.end_cell_id = useful_cells[0];
  current_chunk.start_particle_idx = sim->cell_part_starts[useful_cells[0]];
  current_chunk.particle_count = sim->cell_part_counts[useful_cells[0]];
  current_chunk.grid_point_count = cells[useful_cells[0]].grid_points.size();

  for (size_t i = 1; i < useful_cells.size(); i++) {
    size_t cid = useful_cells[i];
    size_t prev_cid = useful_cells[i - 1];

    // Check if particles are contiguous in file (cast to size_t for comparison)
    size_t expected_next_idx =
        static_cast<size_t>(sim->cell_part_starts[prev_cid]) +
        static_cast<size_t>(sim->cell_part_counts[prev_cid]);

    if (static_cast<size_t>(sim->cell_part_starts[cid]) == expected_next_idx) {
      // Extend current chunk
      current_chunk.end_cell_id = cid;
      current_chunk.particle_count += sim->cell_part_counts[cid];
      current_chunk.grid_point_count += cells[cid].grid_points.size();
    } else {
      // Gap found - finalize current chunk and start new one
      chunks.push_back(current_chunk);

      current_chunk.start_cell_id = cid;
      current_chunk.end_cell_id = cid;
      current_chunk.start_particle_idx = sim->cell_part_starts[cid];
      current_chunk.particle_count = sim->cell_part_counts[cid];
      current_chunk.grid_point_count = cells[cid].grid_points.size();
    }
  }
  chunks.push_back(current_chunk); // Don't forget last chunk

  message("Created %zu contiguous chunks from useful cells", chunks.size());

  // Report statistics
  if (chunks.size() > 1) {
    size_t total_useful_particles = 0;
    for (const auto &chunk : chunks) {
      total_useful_particles += chunk.particle_count;
    }
    double efficiency =
        100.0 * total_useful_particles / (double)sim->nr_dark_matter;
    message("Useful cells contain %.1f%% of total particles", efficiency);
  }

  toc("Preparing to read particles");
  return chunks;
}

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
void flagProxyCells(Simulation *sim) {

  tic();

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;
  const int size = metadata->size;

  // Clear any existing proxy information
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    sim->cells[cid].send_ranks.clear();
  }

  // Loop over all cells
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // If this cell is not local, skip it
    if (cell->rank != rank)
      continue;

    // Loop over the neighbours of this cell
    for (Cell *neighbour : cell->neighbours) {

      // If the neighbour is local theres nothing to do
      if (neighbour->rank == rank)
        continue;

      // If the neighbour is not local we need to recieve from it and send to it

      // Flag this cell as a proxy cell
      neighbour->is_proxy = true;

      // Add this rank to the send ranks if its not already there
      bool already_sent = false;
      for (int send_rank : cell->send_ranks) {
        if (send_rank == neighbour->rank) {
          already_sent = true;
          break;
        }
      }
      if (!already_sent) {
        // Add the neighbour's rank to the send ranks
        cell->send_ranks.push_back(neighbour->rank);
      }
    }
  }

  // Count how many cells we will be sending and how many we will be receiving
  int total_send_cells = 0;
  int total_recv_cells = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // Are we receiving from this cell?
    if (cell->is_proxy) {
      total_recv_cells++;
    }

    // Are we sending to any ranks?
    if (!cell->send_ranks.empty()) {
      total_send_cells += cell->send_ranks.size();
    }
  }

  // Barrier for clarity of output
  MPI_Barrier(MPI_COMM_WORLD);

  // Report the number of proxy cells
  message("Sending %d cells (including multiple sends of the same cell to "
          "different ranks)",
          total_send_cells);
  message("Receiving %d cells", total_recv_cells);

#ifdef DEBUGGING_CHECKS
  // Ensure we all agree on what is sent where
  std::vector<int> send_counts(size, 0);
  std::vector<int> recv_counts(size, 0);
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];
    if (cell->is_proxy && cell->rank == rank) {
      error("Found a local proxy cell! (cell %zu)", cid);
    }
    if (cell->is_proxy) {
      recv_counts[cell->rank]++;
    }
    for (int send_rank : cell->send_ranks) {
      send_counts[send_rank]++;
    }
  }
  // Compare each ranks sends and recvs to make sure everyone agrees
  for (int this_rank = 0; this_rank < size; this_rank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (this_rank != rank)
      continue;
    for (int r = 0; r < size; r++) {
      message("Sending %d cells to rank %d", send_counts[r], r);
      message("Receiving %d cells from rank %d", recv_counts[r], r);
    }
  }
#endif
  toc("Flagging proxy cells");
}
#endif

#ifdef WITH_MPI
void exchangeProxyCells(Simulation *sim) {

  tic();

  // Get the metadata and MPI info
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;

  // Set up vectors to hold MPI requests and receive buffers
  std::vector<MPI_Request> requests;
  std::vector<std::vector<double>> send_buffers;
  std::vector<std::vector<double>> recv_buffers;
  std::vector<size_t> recv_cell_ids;

  // Post non-blocking sends for all local cells that need to be sent
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // Skip if no sends
    if (cell->send_ranks.empty())
      continue;

    // Loop over the ranks we are sending to
    for (int dest_rank : cell->send_ranks) {
      MPI_Request req;

      // Send particle data (mass + positions for each particle)
      send_buffers.emplace_back();
      std::vector<double> &particle_data = send_buffers.back();
      for (size_t p = 0; p < cell->part_count; p++) {
        particle_data.push_back(cell->particles[p]->mass);
        particle_data.push_back(cell->particles[p]->pos[0]);
        particle_data.push_back(cell->particles[p]->pos[1]);
        particle_data.push_back(cell->particles[p]->pos[2]);
      }

      // Post the send
      MPI_Isend(particle_data.data(), particle_data.size(), MPI_DOUBLE,
                dest_rank, cid, MPI_COMM_WORLD, &req);
      requests.push_back(req);
    }
  }

  // Post non-blocking receives for proxy cells
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // Skip if not a proxy cell
    if (!cell->is_proxy)
      continue;

    // Get the source rank for this cell
    int src_rank = cell->rank;

    MPI_Request req;

    // Prepare to receive particle data (mass + positions for each particle)
    recv_buffers.emplace_back(cell->part_count *
                              4); // mass + 3 positions per particle
    recv_cell_ids.push_back(cid);

    MPI_Irecv(recv_buffers.back().data(), recv_buffers.back().size(),
              MPI_DOUBLE, src_rank, cid, MPI_COMM_WORLD, &req);

    requests.push_back(req);
  }

  message("Posted %d operations", (int)requests.size());

  // Wait for all operations to complete
  if (!requests.empty()) {
    message("Waiting for %zu requests", requests.size());
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  // Create particles for proxy cells using the received data
  for (size_t i = 0; i < recv_buffers.size(); i++) {
    size_t cid = recv_cell_ids[i];
    Cell *cell = &sim->cells[cid];
    const std::vector<double> &recv_particle_data = recv_buffers[i];

    // Reserve space for particles in the cell
    try {
      cell->particles.reserve(recv_particle_data.size() / 4);
    } catch (const std::bad_alloc &e) {
      error("Memory allocation failed while reserving space for particles in "
            "cell %zu. System out of memory. Error: %s",
            cid, e.what());
    }

    // Data was received in the order: mass, pos[0], pos[1], pos[2] per particle
    for (size_t p = 0; p < cell->part_count; p++) {
      double mass = recv_particle_data[p * 4];
      double pos[3] = {recv_particle_data[p * 4 + 1],
                       recv_particle_data[p * 4 + 2],
                       recv_particle_data[p * 4 + 3]};

      Particle *part = new Particle(pos, mass);
      cell->particles.push_back(part);
      cell->mass += mass;
    }
  }

  message("Rank %d: Proxy cell exchange completed", rank);
  toc("Exchanging proxy cells");
}

/**
 * @brief Partition chunks across ranks for balanced reading
 *
 * Assigns each chunk to an MPI rank such that each rank reads approximately
 * the same number of particles. This balances I/O workload while maintaining
 * contiguous reads.
 *
 * @param chunks Vector of ParticleChunk structs to assign
 * @param size Number of MPI ranks
 */
void partitionChunksForReading(std::vector<ParticleChunk> &chunks, int size) {
  tic();

  if (size <= 1) {
    // Single rank - assign all chunks to rank 0
    for (auto &chunk : chunks) {
      chunk.reading_rank = 0;
    }
    toc("Partitioning chunks for reading");
    return;
  }

  // Calculate total particles across all chunks
  size_t total_particles = 0;
  for (const auto &chunk : chunks) {
    total_particles += chunk.particle_count;
  }

  size_t particles_per_rank = total_particles / size;

  message("Assigning %zu chunks across %d ranks (target: %zu particles/rank)",
          chunks.size(), size, particles_per_rank);

  // Assign chunks to ranks to balance read workload
  int current_rank = 0;
  size_t rank_particle_count = 0;

  for (auto &chunk : chunks) {
    chunk.reading_rank = current_rank;
    rank_particle_count += chunk.particle_count;

    // Move to next rank if this one has enough particles
    if (rank_particle_count >= particles_per_rank && current_rank < size - 1) {
      message("Rank %d will read %zu particles", current_rank,
              rank_particle_count);
      current_rank++;
      rank_particle_count = 0;
    }
  }

  // Report final rank assignment
  if (current_rank == size - 1 && rank_particle_count > 0) {
    message("Rank %d will read %zu particles", current_rank,
            rank_particle_count);
  }

  toc("Partitioning chunks for reading");
}

/**
 * @brief Partition cells by computational work for load balancing
 *
 * After particles are read, this function redistributes cells across ranks
 * based on computational cost = n_particles * n_grid_points * n_kernels.
 * This ensures balanced computation during kernel mass calculations.
 *
 * @param sim The simulation object
 * @param grid The grid object containing kernel information
 */
void partitionCellsByWork(Simulation *sim, Grid *grid) {
  tic();

  Metadata *metadata = &Metadata::getInstance();
  std::vector<Cell> &cells = sim->cells;

  const int rank = metadata->rank;
  const int size = metadata->size;

  if (size <= 1) {
    // Single rank - assign all cells to rank 0
    metadata->local_work_cost = 0;
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      if (cells[cid].is_useful) {
        cells[cid].rank = 0;
        size_t cost = cells[cid].part_count * cells[cid].grid_points.size() *
                      grid->nkernels;
        metadata->local_work_cost += cost;
      }
    }
    toc("Partitioning cells by work");
    return;
  }

  // Calculate work cost for each useful cell
  // cost = n_particles * n_grid_points * n_kernel_radii
  size_t n_kernels = grid->nkernels;

  std::vector<size_t> useful_cell_ids;
  std::vector<size_t> cell_costs;
  size_t total_cost = 0;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (!cells[cid].is_useful)
      continue;

    size_t cost =
        cells[cid].part_count * cells[cid].grid_points.size() * n_kernels;

    useful_cell_ids.push_back(cid);
    cell_costs.push_back(cost);
    total_cost += cost;
  }

  message("Total computational cost across all useful cells: %zu", total_cost);

  // Partition to balance total cost across ranks
  size_t cost_per_rank = total_cost / size;

  message("Target cost per rank: %zu", cost_per_rank);

  int current_rank = 0;
  size_t rank_cost = 0;

  for (size_t i = 0; i < useful_cell_ids.size(); i++) {
    size_t cid = useful_cell_ids[i];
    cells[cid].rank = current_rank;
    rank_cost += cell_costs[i];

    // Move to next rank if target reached and not last rank
    if (rank_cost >= cost_per_rank && current_rank < size - 1) {
      message("Assigned cost %zu to rank %d", rank_cost, current_rank);
      current_rank++;
      rank_cost = 0;
    }
  }

  // Report final rank assignment
  if (current_rank == size - 1 && rank_cost > 0) {
    message("Assigned cost %zu to rank %d", rank_cost, current_rank);
  }

  // Track local work for this rank
  metadata->local_work_cost = 0;
  int nr_local_useful_cells = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].rank == rank && cells[cid].is_useful) {
      size_t cost =
          cells[cid].part_count * cells[cid].grid_points.size() * n_kernels;
      metadata->local_work_cost += cost;
      nr_local_useful_cells++;
    }
  }

  message("Rank %d: assigned %d useful cells with total cost %zu", rank,
          nr_local_useful_cells, metadata->local_work_cost);

  toc("Partitioning cells by work");
}

/**
 * @brief Redistribute particles from reading ranks to compute ranks
 *
 * This function moves particles from the ranks that read them (based on I/O
 * partitioning) to the ranks that will compute with them (based on work
 * partitioning). After this function, each rank has particles for all cells
 * it owns according to the work partition.
 *
 * @param sim The simulation object
 * @param chunks Vector of ParticleChunk structs containing read particle data
 */
void redistributeParticles(Simulation *sim, std::vector<ParticleChunk> &chunks) {
  tic();

  Metadata *metadata = &Metadata::getInstance();
  std::vector<Cell> &cells = sim->cells;
  const int rank = metadata->rank;
  const int size = metadata->size;

  if (size <= 1) {
    // Single rank - just attach particles to cells directly
    for (auto &chunk : chunks) {
      size_t particle_offset = 0;
      for (size_t cid = chunk.start_cell_id; cid <= chunk.end_cell_id; cid++) {
        if (!cells[cid].is_useful)
          continue;

        size_t npart = cells[cid].part_count;
        for (size_t p = 0; p < npart; p++) {
          cells[cid].particles.push_back(
              new Particle(chunk.positions[particle_offset + p],
                           chunk.masses[particle_offset + p]));
        }

        // Calculate cell mass as sum of all particle masses in this cell
        cells[cid].mass = 0.0;
        if (cells[cid].part_count > 0) {
          for (size_t p = 0; p < npart; p++) {
            cells[cid].mass += chunk.masses[particle_offset + p];
          }
        }

        particle_offset += npart;
      }
    }
    toc("Redistributing particles (serial - no MPI)");
    return;
  }

  message("Rank %d: Starting particle redistribution for work balance", rank);

  int nr_sends = 0;
  int nr_receives = 0;

  for (auto &chunk : chunks) {
    int reading_rank = chunk.reading_rank;

    size_t particle_offset = 0;
    for (size_t cid = chunk.start_cell_id; cid <= chunk.end_cell_id; cid++) {
      if (!cells[cid].is_useful)
        continue;

      int final_rank = cells[cid].rank;
      size_t npart = cells[cid].part_count;

      if (reading_rank == final_rank) {
        // Already on correct rank - attach particles directly
        if (rank == reading_rank) {
          for (size_t p = 0; p < npart; p++) {
            cells[cid].particles.push_back(
                new Particle(chunk.positions[particle_offset + p],
                             chunk.masses[particle_offset + p]));
          }
          // Calculate cell mass as sum of particle masses
          cells[cid].mass = 0.0;
          for (const auto *part : cells[cid].particles) {
            cells[cid].mass += part->mass;
          }
        }
      } else {
        // Need MPI transfer
        if (rank == reading_rank) {
          // Send particles to final_rank
          // Tag = cell ID to match sends/receives
          MPI_Send(&npart, 1, MPI_UNSIGNED_LONG, final_rank, (int)cid,
                   MPI_COMM_WORLD);

          // Send masses
          MPI_Send(chunk.masses.data() + particle_offset, (int)npart, MPI_DOUBLE,
                   final_rank, (int)cid + 1000000, MPI_COMM_WORLD);

          // Send positions (3 doubles per particle)
          MPI_Send(chunk.positions.data() + particle_offset, (int)npart * 3,
                   MPI_DOUBLE, final_rank, (int)cid + 2000000, MPI_COMM_WORLD);

          nr_sends++;
        }

        if (rank == final_rank) {
          // Receive particles
          size_t recv_count;
          MPI_Status status;
          MPI_Recv(&recv_count, 1, MPI_UNSIGNED_LONG, reading_rank, (int)cid,
                   MPI_COMM_WORLD, &status);

          std::vector<double> recv_masses(recv_count);
          std::vector<std::array<double, 3>> recv_positions(recv_count);

          MPI_Recv(recv_masses.data(), (int)recv_count, MPI_DOUBLE, reading_rank,
                   (int)cid + 1000000, MPI_COMM_WORLD, &status);

          MPI_Recv(recv_positions.data(), (int)recv_count * 3, MPI_DOUBLE,
                   reading_rank, (int)cid + 2000000, MPI_COMM_WORLD, &status);

          // Attach to cell
          cells[cid].mass = 0.0;
          for (size_t p = 0; p < recv_count; p++) {
            Particle *part = new Particle(recv_positions[p], recv_masses[p]);
            cells[cid].particles.push_back(part);
            cells[cid].mass += part->mass;
          }

          nr_receives++;
        }
      }

      particle_offset += npart;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  message("Rank %d: Particle redistribution complete (sent %d, received %d cells)",
          rank, nr_sends, nr_receives);

  // Clear chunk data to free memory
  for (auto &chunk : chunks) {
    chunk.masses.clear();
    chunk.positions.clear();
    chunk.masses.shrink_to_fit();
    chunk.positions.shrink_to_fit();
  }

  toc("Redistributing particles");
}

/**
 * @brief Exchange proxy cells between neighboring ranks
 *
 * This function sends copies of boundary cells to neighboring ranks that
 * need them for kernel computations. After this function, each rank has:
 * - Particles for all cells it owns (from work partitioning)
 * - Particles for all proxy cells (neighbors within kernel radius)
 *
 * @param sim The simulation object
 * @param chunks Vector of ParticleChunk structs (unused but kept for consistency)
 */
void exchangeProxyCells(Simulation *sim, std::vector<ParticleChunk> &chunks) {
  tic();

  Metadata *metadata = &Metadata::getInstance();
  std::vector<Cell> &cells = sim->cells;
  const int rank = metadata->rank;
  const int size = metadata->size;

  if (size <= 1) {
    toc("Exchanging proxy cells (serial - no MPI)");
    return;
  }

  message("Rank %d: Starting proxy cell exchange", rank);

  int nr_proxy_sends = 0;
  int nr_proxy_receives = 0;

  // Structure to hold send data (must persist until MPI_Waitall)
  struct SendData {
    size_t npart;
    std::vector<double> masses;
    std::vector<std::array<double, 3>> positions;
    MPI_Request req_count;
    MPI_Request req_masses;
    MPI_Request req_positions;
  };

  // Structure to hold receive data
  struct RecvData {
    size_t cid;
    size_t npart;
    std::vector<double> masses;
    std::vector<std::array<double, 3>> positions;
    MPI_Request req_count;
    MPI_Request req_masses;
    MPI_Request req_positions;
  };

  std::vector<SendData> send_buffers;
  std::vector<RecvData> recv_buffers;

  // Phase 1: Post all non-blocking receives first
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    // Receive cells we need as proxy (using is_proxy flag set by flagProxyCells)
    if (cell->rank != rank && cell->is_proxy) {
      int owner_rank = cell->rank;
      int tag_base = 10000000 + (int)(cid % 1000000); // Safe tag modulo

      RecvData recv_data;
      recv_data.cid = cid;

      // Post receive for particle count
      MPI_Irecv(&recv_data.npart, 1, MPI_UNSIGNED_LONG, owner_rank, tag_base,
                MPI_COMM_WORLD, &recv_data.req_count);

      recv_buffers.push_back(std::move(recv_data));
      nr_proxy_receives++;
    }
  }

  // Phase 2: Post all non-blocking sends
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    // Send this cell to ranks that need it as proxy
    if (cell->rank == rank && !cell->send_ranks.empty()) {
      size_t npart = cell->particles.size();

      // Pack particle data
      for (int dest_rank : cell->send_ranks) {
        int tag_base = 10000000 + (int)(cid % 1000000); // Safe tag modulo

        SendData send_data;
        send_data.npart = npart;
        send_data.masses.resize(npart);
        send_data.positions.resize(npart);

        for (size_t p = 0; p < npart; p++) {
          send_data.masses[p] = cell->particles[p]->mass;
          send_data.positions[p] = cell->particles[p]->pos;
        }

        // Post sends
        MPI_Isend(&send_data.npart, 1, MPI_UNSIGNED_LONG, dest_rank, tag_base,
                  MPI_COMM_WORLD, &send_data.req_count);
        MPI_Isend(send_data.masses.data(), (int)npart, MPI_DOUBLE, dest_rank,
                  tag_base + 1000000, MPI_COMM_WORLD, &send_data.req_masses);
        MPI_Isend(send_data.positions.data(), (int)npart * 3, MPI_DOUBLE,
                  dest_rank, tag_base + 2000000, MPI_COMM_WORLD,
                  &send_data.req_positions);

        send_buffers.push_back(std::move(send_data));
        nr_proxy_sends++;
      }
    }
  }

  // Phase 3: Wait for count receives, then post data receives
  for (auto &recv_data : recv_buffers) {
    MPI_Wait(&recv_data.req_count, MPI_STATUS_IGNORE);

    // Now we know how many particles to receive
    recv_data.masses.resize(recv_data.npart);
    recv_data.positions.resize(recv_data.npart);

    Cell *cell = &cells[recv_data.cid];
    int owner_rank = cell->rank;
    int tag_base = 10000000 + (int)(recv_data.cid % 1000000);

    MPI_Irecv(recv_data.masses.data(), (int)recv_data.npart, MPI_DOUBLE,
              owner_rank, tag_base + 1000000, MPI_COMM_WORLD,
              &recv_data.req_masses);
    MPI_Irecv(recv_data.positions.data(), (int)recv_data.npart * 3, MPI_DOUBLE,
              owner_rank, tag_base + 2000000, MPI_COMM_WORLD,
              &recv_data.req_positions);
  }

  // Phase 4: Wait for all sends to complete
  for (auto &send_data : send_buffers) {
    MPI_Wait(&send_data.req_count, MPI_STATUS_IGNORE);
    MPI_Wait(&send_data.req_masses, MPI_STATUS_IGNORE);
    MPI_Wait(&send_data.req_positions, MPI_STATUS_IGNORE);
  }

  // Phase 5: Wait for all data receives and attach particles
  for (auto &recv_data : recv_buffers) {
    MPI_Wait(&recv_data.req_masses, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_data.req_positions, MPI_STATUS_IGNORE);

    Cell *cell = &cells[recv_data.cid];
    cell->mass = 0.0;
    for (size_t p = 0; p < recv_data.npart; p++) {
      Particle *part = new Particle(recv_data.positions[p], recv_data.masses[p]);
      cell->particles.push_back(part);
      cell->mass += part->mass;
    }
    cell->is_proxy = true;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  message("Rank %d: Proxy exchange complete (sent %d, received %d proxies)",
          rank, nr_proxy_sends, nr_proxy_receives);

  toc("Exchanging proxy cells");
}
#endif
