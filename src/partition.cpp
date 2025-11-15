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

  // Check for pathological case: too few particles for MPI
  if (sim->nr_dark_matter < static_cast<size_t>(size)) {
    error("Insufficient particles for MPI partitioning: %zu particles across "
          "%d ranks.\n"
          "This simulation should be run in serial mode instead.\n"
          "MPI is only beneficial when particle count >> rank count.",
          sim->nr_dark_matter, size);
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
 * This function determines the optimal strategy for reading particles:
 * - If >75% of cells are useful: return empty vector (signal full read)
 * - If <25% of cells are useful: create chunks with gap filling
 *
 * Gap filling: Merge chunks separated by small gaps (<1% of particles)
 * to reduce HDF5 I/O overhead.
 *
 * Works in both serial and MPI modes.
 *
 * @param sim The simulation object
 * @return Vector of ParticleChunk structs (empty if full read recommended)
 */
std::vector<ParticleChunk> prepareToReadParts(Simulation *sim) {
  tic();

  std::vector<Cell> &cells = sim->cells;

#ifdef WITH_MPI
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;

  // Count useful cells on this rank
  size_t nr_useful_local = 0;
  size_t nr_local = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].rank == rank) {
      nr_local++;
      if (cells[cid].is_useful) {
        nr_useful_local++;
      }
    }
  }

  if (nr_local == 0) {
    message("No cells assigned to this rank");
    toc("Preparing to read particles");
    return {};
  }

  double useful_fraction = (double)nr_useful_local / (double)nr_local;

  // If >75% are useful, use full read (return empty vector as signal)
  if (useful_fraction > 0.75) {
    message("%.1f%% of local cells are useful - using full read strategy",
            useful_fraction * 100.0);
    toc("Preparing to read particles");
    return {};
  }

  message("%.1f%% of local cells are useful - using chunked read strategy",
          useful_fraction * 100.0);

  // Build sorted list of useful cell IDs for this rank
  std::vector<size_t> useful_cells;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].rank == rank && cells[cid].is_useful) {
      useful_cells.push_back(cid);
    }
  }
#else
  // Serial mode - check fraction of all cells
  size_t nr_useful = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].is_useful) {
      nr_useful++;
    }
  }

  if (nr_useful == 0) {
    message("No useful cells found - nothing to read");
    toc("Preparing to read particles");
    return {};
  }

  double useful_fraction = (double)nr_useful / (double)sim->nr_cells;

  // If >75% are useful, use full read
  if (useful_fraction > 0.75) {
    message("%.1f%% of cells are useful - using full read strategy",
            useful_fraction * 100.0);
    toc("Preparing to read particles");
    return {};
  }

  message("%.1f%% of cells are useful - using chunked read strategy",
          useful_fraction * 100.0);

  // Build sorted list of useful cell IDs
  std::vector<size_t> useful_cells;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].is_useful) {
      useful_cells.push_back(cid);
    }
  }
#endif

  if (useful_cells.empty()) {
    toc("Preparing to read particles");
    return {};
  }

  // Build initial chunks where consecutive cells have contiguous particles
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

    // Check if particles are contiguous in file
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
  chunks.push_back(current_chunk);

  message("Created %zu initial chunks from useful cells", chunks.size());

  // Gap filling: merge chunks with small gaps between them
  double gap_fraction = metadata->gap_fill_fraction;
  const size_t gap_threshold =
      static_cast<size_t>(gap_fraction * sim->nr_dark_matter);

  std::vector<ParticleChunk> merged_chunks;
  if (!chunks.empty()) {
    merged_chunks.push_back(chunks[0]);

    for (size_t i = 1; i < chunks.size(); i++) {
      ParticleChunk &last_merged = merged_chunks.back();
      const ParticleChunk &current = chunks[i];

      // Calculate gap size in particles
      size_t last_end_idx =
          last_merged.start_particle_idx + last_merged.particle_count;
      size_t gap_size = current.start_particle_idx - last_end_idx;

      if (gap_size < gap_threshold) {
        // Merge: extend last chunk to include gap and current chunk
        last_merged.end_cell_id = current.end_cell_id;
        last_merged.particle_count =
            (current.start_particle_idx + current.particle_count) -
            last_merged.start_particle_idx;
        last_merged.grid_point_count += current.grid_point_count;
      } else {
        // Gap too large - keep as separate chunk
        merged_chunks.push_back(current);
      }
    }
  }

  message("After gap filling: %zu chunks", merged_chunks.size());

  // Report statistics
  size_t total_chunk_particles = 0;
  for (const auto &chunk : merged_chunks) {
    total_chunk_particles += chunk.particle_count;
  }

#ifdef WITH_MPI
  double efficiency =
      100.0 * total_chunk_particles / (double)metadata->nr_local_particles;
  message("Chunks contain %.1f%% of local particles (including gaps)",
          efficiency);
#else
  double efficiency =
      100.0 * total_chunk_particles / (double)sim->nr_dark_matter;
  message("Chunks contain %.1f%% of total particles (including gaps)",
          efficiency);
#endif

  toc("Preparing to read particles");
  return merged_chunks;
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

  // Step 1: Exchange particle counts first
  // After checkAndMoveParticles, part_count may have changed on owning ranks
  // We need to communicate these updated counts before exchanging particles
  std::vector<size_t> local_part_counts(sim->nr_cells);
  std::vector<size_t> global_part_counts(sim->nr_cells);

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    local_part_counts[cid] = sim->cells[cid].part_count;
  }

  // Use Allreduce with MAX to get the actual counts from owning ranks
  // (only the owning rank will have the correct count)
  MPI_Allreduce(local_part_counts.data(), global_part_counts.data(),
                sim->nr_cells, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

  // Update local part_counts with the global values
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    sim->cells[cid].part_count = global_part_counts[cid];
  }

  // Step 2: Post non-blocking sends for all local cells that need to be sent
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
      // Use particles.size() instead of part_count to handle ranks with no
      // local particles
      for (size_t p = 0; p < cell->particles.size(); p++) {
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
    // Use actual received data size instead of part_count to handle ranks with
    // no local particles
    size_t num_particles = recv_particle_data.size() / 4;
    for (size_t p = 0; p < num_particles; p++) {
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
#endif
