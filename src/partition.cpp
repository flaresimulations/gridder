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
      cell_ranks[cid] = cell->rank;
    }
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
        // Flag this cell is a proxy cell
        cell->is_proxy = true;

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
    if (cell->is_proxy) {
      total_recv_cells++;
    }

    // Are we sending to any ranks?
    if (!cell->send_ranks.empty()) {
      total_send_cells += cell->send_ranks.size();
    }
  }

  // Report the number of proxy cells
  message("Sending %d cells (including multiple sends of the same cell to "
          "different ranks)",
          total_send_cells);
  message("Receiving %d cells", total_recv_cells);
}
#endif

#ifdef WITH_MPI
void exchangeProxyCells(Simulation *sim) {
  // Get the metadata and MPI info
  Metadata *metadata = &Metadata::getInstance();
  const int rank = metadata->rank;

  // Set up a vector to hold MPI requests
  std::vector<MPI_Request> requests;

  // Post non-blocking sends for all local cells that need to be sent
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    // Skip if foreign or not a proxy cell
    if (cell->rank != rank || !cell->is_proxy)
      continue;

    // Loop over the ranks we are sending to
    for (int dest_rank : cell->send_ranks) {
      MPI_Request req;

      // Send particle data (mass + positions for each particle)
      std::vector<double> particle_data;
      for (int p = 0; p < cell->part_count; p++) {
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
    std::vector<double> recv_particle_data(
        cell->part_count * 4); // mass + 3 positions per particle
    MPI_Irecv(recv_particle_data.data(), recv_particle_data.size(), MPI_DOUBLE,
              src_rank, cid, MPI_COMM_WORLD, &req);

    requests.push_back(req);
  }

  message("Posted %d operations", rank, (int)requests.size());

  // Wait for all operations to complete
  if (!requests.empty()) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  // Create particles for proxy cells
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &sim->cells[cid];

    if (!cell->is_proxy)
      continue;

    // Extract received particle data and create particles
    std::vector<double> recv_particle_data(cell->part_count * 4);

    // Data was received in the order we posted receives
    for (int p = 0; p < cell->part_count; p++) {
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
}
#endif
