// Standard includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "hdf_io.hpp"
#include "metadata.hpp"
#include "particle.hpp"
#include "simulation.hpp"

/**
 * @brief Is this cell within a grid point's kernel radius?
 *
 * @param grid_point The grid point to check
 * @param kernel_rad2 The squared kernel radius
 *
 * @return True if the cell is within the kernel radius of the grid point,
 * false otherwise.
 */
bool Cell::inKernel(const GridPoint *grid_point,
                    const double kernel_rad2) const {

  // Get the boxsize from the metadata
  Metadata *metadata = &Metadata::getInstance();
  double *dim = metadata->sim->dim;

  // Get the minimum and maximum positions for the cell.
  const double thisx_min = this->loc[0];
  const double thisy_min = this->loc[1];
  const double thisz_min = this->loc[2];
  const double thisx_max = this->loc[0] + this->width[0];
  const double thisy_max = this->loc[1] + this->width[1];
  const double thisz_max = this->loc[2] + this->width[2];

  // Get the position of the grid point
  const double gridx = grid_point->loc[0];
  const double gridy = grid_point->loc[1];
  const double gridz = grid_point->loc[2];

  // Get the maximum distance between the particle and the grid point
  const double dx = std::max({fabs(nearest(thisx_min - gridx, dim[0])),
                              fabs(nearest(thisx_max - gridx, dim[0]))});
  const double dy = std::max({fabs(nearest(thisy_min - gridy, dim[1])),
                              fabs(nearest(thisy_max - gridy, dim[1]))});
  const double dz = std::max({fabs(nearest(thisz_min - gridz, dim[2])),
                              fabs(nearest(thisz_max - gridz, dim[2]))});
  const double r2 = dx * dx + dy * dy + dz * dz;

  return r2 <= kernel_rad2;
}

/**
 * @brief Is this cell entirely outside a grid point's kernel radius?
 *
 * @param grid_point The grid point to check
 * @param kernel_rad2 The squared kernel radius
 *
 * @return True if the cell is outside the kernel radius of the grid point,
 * false otherwise.
 */
bool Cell::outsideKernel(const GridPoint *grid_point,
                         const double kernel_rad2) const {

  // Get the boxsize from the metadata
  Metadata *metadata = &Metadata::getInstance();
  double *dim = metadata->sim->dim;

  // Get cell centre and diagonal
  double cell_centre[3] = {this->loc[0] + this->width[0] / 2.0,
                           this->loc[1] + this->width[1] / 2.0,
                           this->loc[2] + this->width[2] / 2.0};
  double diag2 = this->width[0] * this->width[0] +
                 this->width[1] * this->width[1] +
                 this->width[2] * this->width[2];

  // Get the distance between the grid point and the cell centre
  double dx = nearest(grid_point->loc[0] - cell_centre[0], dim[0]);
  if (fabs(dx) < this->width[0] / 2.0)
    dx = 0.0;
  double dy = nearest(grid_point->loc[1] - cell_centre[1], dim[1]);
  if (fabs(dy) < this->width[1] / 2.0)
    dy = 0.0;
  double dz = nearest(grid_point->loc[2] - cell_centre[2], dim[2]);
  if (fabs(dz) < this->width[2] / 2.0)
    dz = 0.0;
  double r2 = dx * dx + dy * dy + dz * dz;
  r2 -= 1.1 * diag2; // Add a little bit of padding

#ifdef DEBUGGING_CHECKS
  // Ensure we aren't reporting we're outside when particles are inside
  if (r2 > kernel_rad2) {
    for (size_t p = 0; p < this->part_count; p++) {
      Particle *part = this->particles[p];
      const double p_dx = nearest(part->pos[0] - grid_point->loc[0], dim[0]);
      const double p_dy = nearest(part->pos[1] - grid_point->loc[1], dim[1]);
      const double p_dz = nearest(part->pos[2] - grid_point->loc[2], dim[2]);
      const double p_r2 = p_dx * p_dx + p_dy * p_dy + p_dz * p_dz;
      if (p_r2 <= kernel_rad2) {
        error("Particle inside kernel radius but cell outside (dx=%f, dy = % f,"
              "dz=%f, r2=%f, part_r2=%f, kernel_rad2 = %f) "
              "(cell->loc = %f %f %f, cell->width = %f %f %f, "
              "grid_point->loc = "
              "%f %f %f part->pos = %f %f %f)",
              dx, dy, dz, r2, p_r2, kernel_rad2, this->loc[0], this->loc[1],
              this->loc[2], this->width[0], this->width[1], this->width[2],
              grid_point->loc[0], grid_point->loc[1], grid_point->loc[2],
              part->pos[0], part->pos[1], part->pos[2]);
      }
    }
  }
#endif

  return r2 > kernel_rad2;
}

/**
 * @brief Split the cell into 8 children.
 */
void Cell::split() {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();
  Simulation *sim = metadata->sim;

  // Update the max depth
  if (this->depth > sim->max_depth)
    sim->max_depth = this->depth;

#ifdef DEBUGGING_CHECKS

  // Ensure all the grid points within this cell are in the correct place
  for (size_t p = 0; p < this->grid_points.size(); p++) {
    if (this->grid_points[p]->loc[0] < this->loc[0] ||
        this->grid_points[p]->loc[0] >= this->loc[0] + this->width[0] ||
        this->grid_points[p]->loc[1] < this->loc[1] ||
        this->grid_points[p]->loc[1] >= this->loc[1] + this->width[1] ||
        this->grid_points[p]->loc[2] < this->loc[2] ||
        this->grid_points[p]->loc[2] >= this->loc[2] + this->width[2])
      error("Grid point not in correct cell");
  }
#endif

  // Calculate the new width of the children
  double new_width[3] = {this->width[0] / 2.0, this->width[1] / 2.0,
                         this->width[2] / 2.0};

  // Check we actually need to split
  if (this->part_count < metadata->max_leaf_count &&
      this->grid_points.size() <= 1) {
    this->is_split = false;
    return;
  }

  // Prevent infinite recursion from co-located particles
  if (this->depth >= Cell::MAX_OCTREE_DEPTH) {
    error("Maximum octree depth (%d) exceeded at cell with %zu particles. "
          "This indicates co-located particles or numerical precision issues. "
          "Cell location: [%f, %f, %f], width: [%f, %f, %f]",
          Cell::MAX_OCTREE_DEPTH, this->part_count, this->loc[0], this->loc[1],
          this->loc[2], this->width[0], this->width[1], this->width[2]);
  }

  // Flag that we are splitting this cell
  this->is_split = true;

  // Loop over the children creating the cells and attaching the particles and
  // grid points
  for (int i = 0; i < OCTREE_DIM; i++) {
    for (int j = 0; j < OCTREE_DIM; j++) {
      for (int k = 0; k < OCTREE_DIM; k++) {
        // Define the index of the child
        int iprogeny = k + OCTREE_DIM * j + OCTREE_DIM * OCTREE_DIM * i;

        // Calculate the new location of the child
        double new_loc[3];
        new_loc[0] = this->loc[0] + i * new_width[0];
        new_loc[1] = this->loc[1] + j * new_width[1];
        new_loc[2] = this->loc[2] + k * new_width[2];

        // Create child cell using raw pointer allocation
        Cell *child = nullptr;
        try {
          child = new Cell(new_loc, new_width, this, this->top);
        } catch (const std::bad_alloc &e) {
          error("Memory allocation failed while creating child cell (depth=%d, "
                "particles=%zu). System out of memory. Try reducing "
                "n_grid_points "
                "or max_leaf_count parameters. Error: %s",
                this->depth, this->part_count, e.what());
        }

#ifdef WITH_MPI
        // Set the rank of the child
        child->rank = this->rank;
#endif

        // Attach the child to this cell
        this->children[iprogeny] = child;
      }
    }
  }

  // Loop over the particles and attach them to the right child
  for (Particle *part : this->particles) {

    // Get the position of the particle
    const double x = part->pos[0];
    const double y = part->pos[1];
    const double z = part->pos[2];

    // Calculate the child index based on the particle position
    int i = (x >= this->loc[0] + new_width[0]) ? 1 : 0;
    int j = (y >= this->loc[1] + new_width[1]) ? 1 : 0;
    int k = (z >= this->loc[2] + new_width[2]) ? 1 : 0;
    int child_index = k + OCTREE_DIM * j + OCTREE_DIM * OCTREE_DIM * i;

    // Attach the particle to the child cell
    Cell *child = this->children[child_index];

    if (child == nullptr) {
      error("Child cell is null at index %d for particle at (%f, %f, %f) in "
            "cell with location (%f, %f, %f) and width (%f, %f, %f)",
            child_index, x, y, z, this->loc[0], this->loc[1], this->loc[2],
            this->width[0], this->width[1], this->width[2]);
    }

    // Add the particle to the child cell
    child->addParticle(part);
  }

  // Loop over the grid points and attach them to the right child
  for (size_t p = 0; p < this->grid_points.size(); p++) {

    // Get the position of the grid point
    const double x = this->grid_points[p]->loc[0];
    const double y = this->grid_points[p]->loc[1];
    const double z = this->grid_points[p]->loc[2];

    // Calculate the child index based on the grid point position
    int i = (x >= this->loc[0] + new_width[0]) ? 1 : 0;
    int j = (y >= this->loc[1] + new_width[1]) ? 1 : 0;
    int k = (z >= this->loc[2] + new_width[2]) ? 1 : 0;
    int child_index = k + OCTREE_DIM * j + OCTREE_DIM * OCTREE_DIM * i;

    // Attach the grid point to the child cell
    Cell *child = this->children[child_index];

    if (child == nullptr) {
      error("Child cell is null at index %d for grid point at (%f, %f, %f) in "
            "cell with location (%f, %f, %f) and width (%f, %f, %f)",
            child_index, x, y, z, this->loc[0], this->loc[1], this->loc[2],
            this->width[0], this->width[1], this->width[2]);
    }

    // Add the grid point to the child cell
    child->addGridPoint(this->grid_points[p]);
  }

  // Loop over the children and recursively split them if they have too many
  for (int i = 0; i < OCTREE_CHILDREN; i++) {
    Cell *child = this->children[i];
    child->split();
  }

#ifdef DEBUGGING_CHECKS
  // Make sure the sum of child particle counts is the same as the parent
  size_t child_part_count = 0;
  for (int i = 0; i < OCTREE_CHILDREN; i++) {
    child_part_count += this->children[i]->part_count;
  }
  if (child_part_count != this->part_count)
    error("Particle count mismatch in cell (child_part_count = %d, "
          "this->part_count = %d)",
          child_part_count, this->part_count);

  // Ensure all particles in this cell should be in this cell
  for (size_t p = 0; p < this->particles.size(); p++) {
    Particle *part = this->particles[p];
    if (part->pos[0] < this->loc[0] ||
        part->pos[0] >= this->loc[0] + this->width[0] ||
        part->pos[1] < this->loc[1] ||
        part->pos[1] >= this->loc[1] + this->width[1] ||
        part->pos[2] < this->loc[2] ||
        part->pos[2] >= this->loc[2] + this->width[2]) {
      error("Particle %zu in cell %d is outside the cell bounds (%f, %f, %f) "
            "with width (%f, %f, %f)",
            p, this->ph_ind, part->pos[0], part->pos[1], part->pos[2],
            this->width[0], this->width[1], this->width[2]);
    }
  }

  // Make sure the sum of the child grid point counts is the same as the
  // parent
  size_t child_grid_point_count = 0;
  for (int i = 0; i < OCTREE_CHILDREN; i++) {
    child_grid_point_count += this->children[i]->grid_points.size();
  }
  if (child_grid_point_count != this->grid_points.size())
    error("Grid point count mismatch in cell %d (child_grid_point_count = "
          "%d, "
          "this->grid_points.size = %d)",
          this->ph_ind, child_grid_point_count, this->grid_points.size());
#endif
}

/**
 * @brief Add a new particle to the cell.
 *
 * @param part The particle to add.
 */
void Cell::addParticle(Particle *part) {

#ifdef DEBUGGING_CHECKS
  // Check if the particle is already in the cell
  for (size_t i = 0; i < this->particles.size(); i++) {
    if (this->particles[i] == part) {
      error("Particle already in cell %d (particle address: %p)", this->ph_ind,
            part);
    }
  }
#endif

  // Add the particle to the cell
  try {
    this->particles.push_back(part);
    this->part_count++;
    this->mass += part->mass;
  } catch (const std::bad_alloc &e) {
    error("Memory allocation failed while adding particle to cell. "
          "System out of memory. Error: %s",
          e.what());
  }

  // Mark the cell as useful
  this->is_useful = true;
}

/**
 * @brief Get the cell containing a position.
 *
 * @param pos The position of the point.
 *
 * @return The cell containing the point.
 */
Cell *getCellContainingPoint(const double pos[3]) {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();
  Simulation *sim = metadata->sim;

  // Get the cell index
  int i = static_cast<int>(std::floor(pos[0] * sim->inv_width[0]));
  int j = static_cast<int>(std::floor(pos[1] * sim->inv_width[1]));
  int k = static_cast<int>(std::floor(pos[2] * sim->inv_width[2]));

  // Get the cell index
  int cid = (i * sim->cdim[1] * sim->cdim[2]) + (j * sim->cdim[2]) + k;

  // Return the cell
  return &sim->cells[cid];
}

/**
 * @brief Get the cell index containing a position.
 *
 * @param pos The position of the point.
 *
 * @return The cell index containing the point.
 */
int getCellIndexContainingPoint(const double pos[3]) {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();
  Simulation *sim = metadata->sim;

  // Get the cell index
  int i = static_cast<int>(std::floor(pos[0] * sim->inv_width[0]));
  int j = static_cast<int>(std::floor(pos[1] * sim->inv_width[1]));
  int k = static_cast<int>(std::floor(pos[2] * sim->inv_width[2]));

  // Get the cell index
  return (i * sim->cdim[1] * sim->cdim[2]) + (j * sim->cdim[2]) + k;
}

/**
 * @brief Assign particles to cells.
 *
 * @param sim The simulation object.
 */
void assignPartsToCells(Simulation *sim) {

  tic();

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Unpack the cell particle counts and offsets
  std::vector<int> &counts = sim->cell_part_counts;
  std::vector<int> &offsets = sim->cell_part_starts;

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Open the HDF5 file
  HDF5Helper hdf(metadata->input_file);

#ifdef WITH_MPI
  // Read the particle data slice for this rank
  std::vector<double> masses;
  std::array<unsigned long long, 1> mass_dims = {
      static_cast<unsigned long long>(metadata->nr_local_particles)};
  std::array<unsigned long long, 1> start_index = {
      static_cast<unsigned long long>(metadata->first_local_part_ind)};
  if (!hdf.readDatasetSlice<double>("PartType1/Masses", masses, start_index,
                                    mass_dims)) {
    error("Failed to read particle masses");
  }

  std::vector<double> poss;
  std::array<unsigned long long, 2> pos_dims = {
      static_cast<unsigned long long>(metadata->nr_local_particles), 3};
  std::array<unsigned long long, 2> pos_start_index = {
      static_cast<unsigned long long>(metadata->first_local_part_ind), 0};
  if (!hdf.readDatasetSlice<double>("PartType1/Coordinates", poss,
                                    pos_start_index, pos_dims)) {
    error("Failed to read particle positions");
  }
#else
  // Read the particle data all at once
  std::vector<double> masses;
  if (!hdf.readDataset<double>(std::string("PartType1/Masses"), masses)) {
    error("Failed to read particle masses");
  }
  std::vector<double> poss;
  if (!hdf.readDataset<double>("PartType1/Coordinates", poss)) {
    error("Failed to read particle positions");
  }
#endif

  // Loop over cells attaching particles and grid points
  size_t total_part_count = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {

    // Get the cell
    Cell *cell = &cells[cid];

    // Skip unuseful cells
    if (!cell->is_useful)
      continue;

#ifdef WITH_MPI
    // Skip if this cell isn't on this rank (proxy cells handled separately)
    if (cell->rank != metadata->rank)
      continue;
#endif

    // Get the particle slice start and length
    size_t offset = offsets[cid];
    size_t count = counts[cid];
    total_part_count += count;

    // Reserve space for the particles in the cell
    try {
      cell->particles.reserve(count);
    } catch (const std::bad_alloc &e) {
      error("Memory allocation failed while reserving space for particles in "
            "cell %zu. System out of memory. Error: %s",
            cid, e.what());
    }

#ifdef WITH_MPI
    // Remove the offset to this rank
    offset -= metadata->first_local_part_ind;
#endif

    // Skip empty cells
    if (count == 0)
      continue;

    // Loop over the particle data making particles
    for (size_t p = offset; p < offset + count; p++) {

      // Get the mass and position of the particle
      const double mass = masses[p];
      const double pos[3] = {poss[p * 3], poss[p * 3 + 1], poss[p * 3 + 2]};

      // Add the mass to the cell
      cell->mass += mass;

      // Attach the particle to the cell
      try {
        cell->particles.push_back(new Particle(pos, mass));
      } catch (const std::bad_alloc &e) {
        error("Memory allocation failed while adding particle to cell %zu "
              "(current size: %zu particles). System out of memory. "
              "Error: %s",
              cid, cell->particles.size(), e.what());
      }

#ifdef DEBUGGING_CHECKS
      // Check the last particle we added is ok
      if (cell->particles.back() == nullptr) {
        error("Failed to add particle to cell %zu (current size: %zu "
              "particles). System out of memory.",
              cid, cell->particles.size());
      }
#endif
    }
  }

  // Compute the total mass in the simulation from the masses vector
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (double mass : masses) {
    total_mass += mass;
  }

#ifdef WITH_MPI
  // Reduce the total mass
  double global_total_mass = 0.0;
  MPI_Allreduce(&total_mass, &global_total_mass, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  total_mass = global_total_mass;
#endif

  // Compute the mean comoving density
  sim->mean_density = total_mass / sim->volume;

  message("Mean comoving density: %e 10**10 Msun / cMpc^3", sim->mean_density);

#ifdef DEBUGGING_CHECKS
  // Make sure we have attached all the particles (only count local cells in
  // MPI)
  size_t total_cell_part_count = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];
#ifdef WITH_MPI
    // Only count particles in local cells for this check
    if (cell->rank == metadata->rank) {
      total_cell_part_count += cell->part_count;
    }
#else
    total_cell_part_count += cell->part_count;
#endif
  }
  if (total_part_count != total_cell_part_count) {
    error("Particle count mismatch (total_part_count = %d, "
          "total_cell_part_count = "
          "%d)",
          total_part_count, total_cell_part_count);
  }
#endif
  toc("Assigning particles to cells");
}

/**
 * @brief Make sure particles are in the right cells and move them if not.
 *
 * This function is the parallel version and will also communicate any
 * particles that need to be moved to other ranks.
 *
 * @param sim The simulation object.
 */
#ifdef WITH_MPI
static void checkAndMoveParticlesMPI(Simulation *sim) {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Initialise a counter for the number of particles moved
  size_t moved_count = 0;

  // Initialise a vector to hold the particles we will send to each rank
  std::vector<std::vector<Particle *>> send_particles(metadata->size);

  // Loop over the cells and check the particles
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {

    // Get the cell
    Cell *cell = &cells[cid];

    // Skip unuseful cells
    if (!cell->is_useful)
      continue;

    // Loop over the particles in this cell
    for (size_t p = 0; p < cell->part_count; p++) {

      // Get the particle
      Particle *part = cell->particles[p];

      // Get the cell containing this particle
      Cell *containing_cell = getCellContainingPoint(part->pos);

      // If the particle is in the right cell, continue
      if (containing_cell == cell)
        continue;

      // Get the rank of the containing cell
      int target_rank = containing_cell->rank;
      if (target_rank < 0 || target_rank >= metadata->size) {
        error("Particle %zu in cell %zu has invalid target rank %d", p, cid,
              target_rank);
      }

      // If the containing cell is local, move the particle to it
      if (containing_cell->rank == metadata->rank) {
        containing_cell->addParticle(part);
        moved_count++;
        continue;
      }

      // Otherwise, store it to send shortly
      try {
        send_particles[target_rank].push_back(part);
      } catch (const std::bad_alloc &e) {
        error("Memory allocation failed while adding particle to send vector "
              "for rank %d. System out of memory. Error: %s",
              target_rank, e.what());
      }
    }
  }

  // We need to tell everyone how many particles to expect from each
  // rank, so we can allocate the right amount of memory for the receives
  std::vector<int> send_counts(metadata->size, 0);
  for (size_t rank = 0; rank < metadata->size; rank++) {
    // Count the number of particles to send to this rank
    send_counts[rank] = static_cast<int>(send_particles[rank].size());
  }

  // Send the counts to all ranks
  std::vector<int> recv_counts(metadata->size * metadata->size, 0);
  for (size_t rank = 0; rank < metadata->size; rank++) {
    // If this is me, tell the recieving ranks how many particles to expect
    if (rank == metadata->rank) {
      MPI_Send(send_counts.data(), metadata->size, MPI_INT, rank, cid,
               MPI_COMM_WORLD);
    } else {
      // Otherwise, receive the counts from the sending rank
      MPI_Recv(recv_counts.data() + rank * metadata->size, metadata->size,
               MPI_INT, rank, cid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Now we need to send and recieve the foreign particles
  std::vector<MPI_Request> requests;
  std::vector<std::vector<double>> recv_buffers;
  for (size_t rank = 0; rank < metadata->size; rank++) {

    // If there are no particles to send to this rank, skip it
    if (send_particles[rank].empty())
      continue;

    // Prepare the data to send
    MPI_Request req;
    std::vector<double> send_buffer.reserve(
        send_particles[rank].size() * 4); // 1 mass + 3 positions per particle
    for (Particle *part : send_particles[rank]) {
      // Add the particle data to the buffer
      send_buffer.push_back(part->mass);
      send_buffer.push_back(part->pos[0]);
      send_buffer.push_back(part->pos[1]);
      send_buffer.push_back(part->pos[2]);
    }

    // Post the send
    MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, rank, cid,
              MPI_COMM_WORLD, &req);
    requests.push_back(req);

#ifdef DEBUGGING_CHECKS
    message("Rellocating %zu particles to rank %d from cell %zu",
            send_particles[rank].size(), rank, cid);
#endif

    // Do we also need to receive particles from this rank?
    if (recv_counts[rank] > 0) {

      // Prepare the receive buffer
      std::vector<double> recv_buffer(recv_counts[rank] * 4); // 1 mass + 3 pos

      // Post the receive
      MPI_Irecv(recv_buffer.data(), recv_counts[rank] * 4, MPI_DOUBLE, rank,
                cid, MPI_COMM_WORLD, &req);
      requests.push_back(req);

      // Store the receive buffer for later processing
      recv_buffers.push_back(std::move(recv_buffer));
    }
  }

  // Wait for all the sends and receives to complete
  if (!requests.empty()) {
    int err =
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    if (err != MPI_SUCCESS) {
      error("MPI_Waitall failed with error code %d", err);
    }
  }

  // Now process the received particles
  for (size_t rank = 0; rank < metadata->size; rank++) {
    // If there are no particles to receive from this rank, skip it
    if (recv_counts[rank] == 0)
      continue;

    // Get the receive buffer for this rank
    std::vector<double> &recv_buffer = recv_buffers[rank];

    // Loop over the received particles
    for (size_t i = 0; i < recv_counts[rank]; i++) {
      // Get the mass and position of the particle
      double mass = recv_buffer[i * 4];
      double pos[3] = {recv_buffer[i * 4 + 1], recv_buffer[i * 4 + 2],
                       recv_buffer[i * 4 + 3]};

      // Create a new particle and add it to the correct cell
      Particle *part = new Particle(mass, pos);
      Cell *containing_cell = getCellContainingPoint(pos);
      containing_cell->addParticle(part);
      moved_count++;
    }
  }

  message("Moved %zu particles to correct cells", moved_count);
}
#endif // WITH_MPI

/**
 * @brief Make sure particles are in the right cells and move them if not.
 *
 * @param sim The simulation object.
 */
void checkAndMoveParticles(Simulation *sim) {

  tic();

#ifdef WITH_MPI
  // If we are using MPI, use the MPI version
  checkAndMoveParticlesMPI(sim);
#else

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Initialise a counter for the number of particles moved
  size_t moved_count = 0;

  // Loop over the cells and check the particles
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {

    // Get the cell
    Cell *cell = &cells[cid];

    // Skip unuseful cells
    if (!cell->is_useful)
      continue;

    // Loop over the particles in this cell
    for (size_t p = 0; p < cell->part_count; p++) {

      // Get the particle
      Particle *part = cell->particles[p];

      // Get the cell containing this particle
      Cell *containing_cell = getCellContainingPoint(part->pos);

      // If the particle is in the right cell, continue
      if (containing_cell == cell)
        continue;

      // Otherwise, move it into the correct cell
      containing_cell->addParticle(part);
      moved_count++;
    }
  }

  message("Moved %zu particles to correct cells", moved_count);

#endif // WITH_MPI

#ifdef DEBUGGING_CHECKS
  // Check that all particles are in the right cells
  if (moved_count > 0) {
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      Cell *cell = &cells[cid];
      for (Particle *part : cell->particles) {
        Cell *containing_cell = getCellContainingPoint(part->pos);
        if (containing_cell != cell) {
          error("Particle at (%f, %f, %f) in cell %zu (%f-%f, %f-%f, %f-%f) "
                "is in the wrong cell",
                part->pos[0], part->pos[1], part->pos[2], cid, cell->loc[0],
                cell->loc[0] + cell->width[0], cell->loc[1],
                cell->loc[1] + cell->width[1], cell->loc[2],
                cell->loc[2] + cell->width[2]);
        }

        if (part->pos[0] < cell->loc[0] ||
            part->pos[0] >= cell->loc[0] + cell->width[0] ||
            part->pos[1] < cell->loc[1] ||
            part->pos[1] >= cell->loc[1] + cell->width[1] ||
            part->pos[2] < cell->loc[2] ||
            part->pos[2] >= cell->loc[2] + cell->width[2]) {
          error("Particle at (%f, %f, %f) in cell %zu is out of bounds",
                part->pos[0], part->pos[1], part->pos[2], cid);
        }
      }
    }
  }
#endif // DEBUGGING_CHECKS

  toc("Moving particles to correct cells");
}

/**
 * @brief Assign grid points to cells.
 *
 * @param cells The cells to assign grid points to.
 */
void assignGridPointsToCells(Simulation *sim, Grid *grid) {

  tic();

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Get the grid points
  std::vector<GridPoint> &grid_points = grid->grid_points;

#pragma omp parallel for
  // Loop over the grid points assigning them to cells
  for (size_t gid = 0; gid < grid_points.size(); gid++) {

    // Get the grid point
    GridPoint *grid_point = &grid_points[gid];

    // Get the cell this grid point is in
    Cell *cell = getCellContainingPoint(grid_point->loc);

    // If the cell is not local, nothing to do
#ifdef WITH_MPI
    // Get the metadata instance for MPI rank checking
    Metadata *metadata = &Metadata::getInstance();
    if (cell->rank != metadata->rank)
      error("Grid point %zu is in cell %zu which is not local to this rank %d",
            gid, getCellIndexContainingPoint(grid_point->loc), metadata->rank);
#endif

#pragma omp critical
    {
      // Attach the grid point to the cell
      cell->grid_points.push_back(grid_point);
    }
  }
#ifdef DEBUGGING_CHECKS

  // Check grid points are in the right cells
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];
    for (GridPoint *grid_point : cell->grid_points) {
      if (grid_point->loc[0] < cell->loc[0] ||
          grid_point->loc[0] >= cell->loc[0] + cell->width[0] ||
          grid_point->loc[1] < cell->loc[1] ||
          grid_point->loc[1] >= cell->loc[1] + cell->width[1] ||
          grid_point->loc[2] < cell->loc[2] ||
          grid_point->loc[2] >= cell->loc[2] + cell->width[2])
        error("Grid point not in correct cell");
    }
  }
#endif

  toc("Assigning grid points to cells");
}

/**
 * @brief Loop over the cells and label the useful ones.
 *
 * A useful cell is one that either contains a grid point or is a neighbour
 * of a cell that contains a grid point.
 *
 * @param sim The simulation object.
 */
void limitToUsefulCells(Simulation *sim) {

  tic();

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Loop over the cells and label the useful ones
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {

    // Get the cell
    Cell *cell = &cells[cid];

    // Check if the cell is useful
    if (cell->grid_points.size() > 0) {
      cell->is_useful = true;
    } else {
      continue;
    }

    // If we got here we have a useful cell, so we need to label the
    // neighbours as useful too
    for (Cell *neighbour : cell->neighbours) {
      neighbour->is_useful = true;
    }
  }

  // Count the number of useful cells
  int useful_count = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].is_useful) {
      useful_count++;
    }
  }

  message("Number of useful cells: %d (out of %d)", useful_count,
          sim->nr_cells);

  toc("Flagging useful cells");
}
