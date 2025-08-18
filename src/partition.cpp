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
    Cell *cell = &sim->cells[cid];

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
  const int rank = metadata->rank;
  const int size = metadata->size;

  // Clear any existing proxy information
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    sim->cells[cid].send_ranks.clear();
    sim->cells[cid].recv_rank = -1;
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
      if (cell->rank != neighbour->rank) {
        // If we are not already receiving from this rank, set it
        if (cell->recv_rank == -1) {
          cell->recv_rank = neighbour->rank;
        } else if (cell->recv_rank != neighbour->rank) {
          error("Cell %zu on rank %d is trying to receive from multiple ranks: "
                "rank %d and rank %d",
                cid, rank, cell->recv_rank, neighbour->rank);
        }

        // Add this rank to the send ranks
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
    if (cell->recv_rank != -1) {
      total_recv_cells++;
    }

    // Are we sending to any ranks?
    if (!cell->send_ranks.empty()) {
      total_send_cells++;
    }
  }

  // Report the number of proxy cells
  message("Sending %d cells", total_send_cells);
  message("Receiving %d cells", total_recv_cells);
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
void packProxyComms(Simulation *sim,
                    std::map<int, std::vector<double>> &rank_masses,
                    std::map<int, std::vector<double>> &rank_positions,
                    std::map<int, std::vector<int>> &rank_particle_cells) {

  // Get the current rank
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // Only send cells that we own and have send ranks
    if (cell->rank != rank || cell->send_ranks.empty())
      continue;

    int npart = cell->part_count;

    // For each rank that needs this cell as a proxy
    for (int dest_rank : cell->send_ranks) {
      // Pack particle data with individual cell assignments
      for (int p = 0; p < npart; p++) {
        Particle *part = cell->particles[p];

        // Calculate which cell this particle belongs to based on position
        int correct_cell = getCellIndexContainingPoint(part->pos);

        rank_masses[dest_rank].push_back(part->mass);
        rank_positions[dest_rank].push_back(part->pos[0]);
        rank_positions[dest_rank].push_back(part->pos[1]);
        rank_positions[dest_rank].push_back(part->pos[2]);
        rank_particle_cells[dest_rank].push_back(correct_cell);
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
void unpackProxyComms(Simulation *sim, const std::vector<double> &masses,
                      const std::vector<double> &positions,
                      const std::vector<int> &particle_cells) {

  // Create particles and assign to correct cells based on particle_cells
  // mapping
  for (size_t p = 0; p < masses.size(); p++) {
    const double mass = masses[p];
    const double pos[3] = {positions[p * 3], positions[p * 3 + 1],
                           positions[p * 3 + 2]};
    int cell_id = particle_cells[p];

    // Create the particle using raw pointer allocation
    Particle *part = new Particle(pos, mass);

    // Get the correct cell and add particle
    Cell *cell = &sim->cells[cell_id];
    cell->mass += mass;
    cell->particles.push_back(part);
    cell->part_count++;
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
  std::map<int, std::vector<int>> rank_particle_cells;

  packProxyComms(sim, rank_masses, rank_positions, rank_particle_cells);

  // Report packing statistics
  int total_send_particles = 0;
  for (const auto &pair : rank_masses) {
    total_send_particles += pair.second.size();
  }
  message("Rank %d: Packed %d particles for sending", rank,
          total_send_particles);

  // Send packed data to each rank (non-blocking)
  std::vector<MPI_Request> requests;
  int send_count = 0;

  for (int dest_rank = 0; dest_rank < size; dest_rank++) {
    if (dest_rank == rank || rank_masses[dest_rank].empty())
      continue;

    int num_particles = rank_masses[dest_rank].size();
    message("Rank %d: Sending %d particles to rank %d", rank, num_particles,
            dest_rank);

    MPI_Request req1, req2, req3, req4;

    // Send particle count first
    MPI_Isend(&num_particles, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, &req1);

    // Send particle data
    MPI_Isend(rank_masses[dest_rank].data(), num_particles, MPI_DOUBLE,
              dest_rank, MPI_TAG_MASS, MPI_COMM_WORLD, &req2);
    MPI_Isend(rank_positions[dest_rank].data(), num_particles * 3, MPI_DOUBLE,
              dest_rank, MPI_TAG_POSITION, MPI_COMM_WORLD, &req3);
    MPI_Isend(rank_particle_cells[dest_rank].data(), num_particles, MPI_INT,
              dest_rank, 3, MPI_COMM_WORLD, &req4);

    requests.insert(requests.end(), {req1, req2, req3, req4});
    send_count++;
  }

  message("Rank %d: Posted sends to %d ranks", rank, send_count);

  // Receive data from each rank
  int recv_count = 0;
  for (int src_rank = 0; src_rank < size; src_rank++) {
    if (src_rank == rank)
      continue;

    // Check if we expect data from this rank
    bool expect_data = false;
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      if (sim->cells[cid].recv_rank == src_rank) {
        expect_data = true;
        break;
      }
    }

    if (!expect_data)
      continue;

    message("Rank %d: Expecting proxy data from rank %d", rank, src_rank);

    // Receive particle count
    int num_particles;
    MPI_Recv(&num_particles, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    message("Rank %d: Receiving %d particles from rank %d", rank, num_particles,
            src_rank);

    // Receive particle data
    std::vector<double> masses(num_particles), positions(num_particles * 3);
    std::vector<int> particle_cells(num_particles);

    MPI_Recv(masses.data(), num_particles, MPI_DOUBLE, src_rank, MPI_TAG_MASS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(positions.data(), num_particles * 3, MPI_DOUBLE, src_rank,
             MPI_TAG_POSITION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(particle_cells.data(), num_particles, MPI_INT, src_rank, 3,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Unpack received data
    unpackProxyComms(sim, masses, positions, particle_cells);
    recv_count++;
  }

  message("Rank %d: Received data from %d ranks", rank, recv_count);

  // Wait for all sends to complete
  if (!requests.empty()) {
    message("Rank %d: Waiting for %zu send operations to complete", rank,
            requests.size());
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  message("Rank %d: Proxy cell exchange completed", rank);
}
#endif
