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
  // Ensure all the particles within this cell are in the correct place
  for (size_t p = 0; p < this->part_count; p++) {
    if (this->particles[p]->pos[0] < this->loc[0] ||
        this->particles[p]->pos[0] >= this->loc[0] + this->width[0] ||
        this->particles[p]->pos[1] < this->loc[1] ||
        this->particles[p]->pos[1] >= this->loc[1] + this->width[1] ||
        this->particles[p]->pos[2] < this->loc[2] ||
        this->particles[p]->pos[2] >= this->loc[2] + this->width[2])
      error("Particle not in correct cell: particle pos=(%f,%f,%f) but cell "
            "bounds=[%f-%f, %f-%f, %f-%f]",
            this->particles[p]->pos[0], this->particles[p]->pos[1],
            this->particles[p]->pos[2], this->loc[0],
            this->loc[0] + this->width[0], this->loc[1],
            this->loc[1] + this->width[1], this->loc[2],
            this->loc[2] + this->width[2]);
  }

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

  // Loop over the children creating the cells and attaching the particles
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

        // Attach the particles to the child and count them while we're at
        // it
        for (size_t p = 0; p < this->part_count; p++) {
          if (this->particles[p]->pos[0] >= new_loc[0] &&
              this->particles[p]->pos[0] < new_loc[0] + new_width[0] &&
              this->particles[p]->pos[1] >= new_loc[1] &&
              this->particles[p]->pos[1] < new_loc[1] + new_width[1] &&
              this->particles[p]->pos[2] >= new_loc[2] &&
              this->particles[p]->pos[2] < new_loc[2] + new_width[2]) {
            try {
              child->particles.push_back(this->particles[p]);
              child->part_count++;
              child->mass += this->particles[p]->mass;
            } catch (const std::bad_alloc &e) {
              error("Memory allocation failed while splitting cell (depth=%d, "
                    "particles=%zu, child_particles=%zu). "
                    "Try reducing n_grid_points or max_leaf_count parameters. "
                    "Error: %s",
                    this->depth, this->part_count, child->part_count, e.what());
            }
          }
        }

        // Attach the grid points to the child
        for (size_t p = 0; p < this->grid_points.size(); p++) {
          if (this->grid_points[p]->loc[0] >= new_loc[0] &&
              this->grid_points[p]->loc[0] < new_loc[0] + new_width[0] &&
              this->grid_points[p]->loc[1] >= new_loc[1] &&
              this->grid_points[p]->loc[1] < new_loc[1] + new_width[1] &&
              this->grid_points[p]->loc[2] >= new_loc[2] &&
              this->grid_points[p]->loc[2] < new_loc[2] + new_width[2]) {
            try {
              child->grid_points.push_back(this->grid_points[p]);
            } catch (const std::bad_alloc &e) {
              error("Memory allocation failed while assigning grid points to "
                    "child cell "
                    "(depth=%d, grid_points=%zu, child_grid_points=%zu). "
                    "Try reducing n_grid_points parameter. Error: %s",
                    this->depth, this->grid_points.size(),
                    child->grid_points.size(), e.what());
            }
          }
        }

        // Attach the child to this cell
        this->children[iprogeny] = child;

        // Split this child
        child->split();
      }
    }
  }

#ifdef DEBUGGING_CHECKS
  // Make sure the sum of child particle counts is the same as the parent
  size_t child_part_count = 0;
  for (int i = 0; i < OCTREE_CHILDREN; i++) {
    child_part_count += this->children[i]->part_count;
  }
  if (child_part_count != this->part_count)
    error("Particle count mismatch in cell %d (child_part_count = %d, "
          "this->part_count = %d)",
          this->ph_ind, child_part_count, this->part_count);

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
  int i = static_cast<int>(pos[0] / sim->width[0]);
  int j = static_cast<int>(pos[1] / sim->width[1]);
  int k = static_cast<int>(pos[2] / sim->width[2]);

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
  int i = static_cast<int>(pos[0] / sim->width[0]);
  int j = static_cast<int>(pos[1] / sim->width[1]);
  int k = static_cast<int>(pos[2] / sim->width[2]);

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

  message("Assigning particles to cells...");

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Unpack the cell particle counts and offsets
  std::vector<int> &counts = sim->cell_part_counts;
  std::vector<int> &offsets = sim->cell_part_starts;

  // Get the cells
  std::vector<Cell> &cells = sim->cells;

  // Open the HDF5 file
  HDF5Helper hdf(metadata->input_file);

  message("Reading particle data from '%s'...", metadata->input_file.c_str());

#ifdef WITH_MPI
  message("Reading particle data (local = %d, total = %d, first_local = %d)...",
          metadata->nr_local_particles, sim->nr_dark_matter,
          metadata->first_local_part_ind);
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
  message("Read %zu particle masses from '%s'", masses.size(),
          metadata->input_file.c_str());

  std::vector<double> poss;
  std::array<unsigned long long, 2> pos_dims = {
      static_cast<unsigned long long>(metadata->nr_local_particles), 3};
  std::array<unsigned long long, 2> pos_start_index = {
      static_cast<unsigned long long>(metadata->first_local_part_ind), 0};
  message("Reading particle positions from '%s'...",
          metadata->input_file.c_str());
  if (!hdf.readDatasetSlice<double>("PartType1/Coordinates", poss,
                                    pos_start_index, pos_dims)) {
    error("Failed to read particle positions");
  }
  message("Read %zu particle positions from '%s'", poss.size() / 3,
          metadata->input_file.c_str());
#else
  // Read the particle data all at once
  std::vector<double> masses;
  message("Reading all particle masses from '%s'...",
          metadata->input_file.c_str());
  if (!hdf.readDataset<double>(std::string("PartType1/Masses"), masses)) {
    error("Failed to read particle masses");
  }
  if (masses.empty()) {
    error("No particle masses found in the dataset");
  }
  message("Read %zu particle masses from '%s'", masses.size(),
          metadata->input_file.c_str());
  std::vector<double> poss;
  message("Reading all particle positions from '%s'...",
          metadata->input_file.c_str());
  if (!hdf.readDataset<double>("PartType1/Coordinates", poss)) {
    error("Failed to read particle positions");
  }
  message("Read %zu particle positions from '%s'", poss.size() / 3,
          metadata->input_file.c_str());
#endif

  message("Read %zu particles from '%s'", masses.size(),
          metadata->input_file.c_str());

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
    int offset = offsets[cid];
    int count = counts[cid];
    total_part_count += count;

#ifdef DEBUGGING_CHECKS
    // Report the number of particles in this cell
    message(
        "Cell %zu has %d particles (offset: %d, count: %d, total_so_far: %zu)",
        cid, count, offset, count, total_part_count);
#endif

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
    for (int p = offset; p < offset + count; p++) {

      // Get the mass and position of the particle
      const double mass = masses[p];
      const double pos[3] = {poss[p * 3], poss[p * 3 + 1], poss[p * 3 + 2]};

      // Create the particle using raw pointer allocation
      Particle *part = nullptr;
      try {
        part = new Particle(pos, mass);
      } catch (const std::bad_alloc &e) {
        error(
            "Memory allocation failed while creating particle %d in cell %zu. "
            "System out of memory. Error: %s",
            p, cid, e.what());
      }

      // Add the mass to the cell
      cell->mass += mass;

      // Attach the particle to the cell
      try {
        cell->particles.push_back(part);
      } catch (const std::bad_alloc &e) {
        error("Memory allocation failed while adding particle to cell %zu "
              "(current size: %zu particles). System out of memory. "
              "Error: %s",
              cid, cell->particles.size(), e.what());
      }
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

  // Check particles are in the right cells
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];
    for (Particle *part : cell->particles) {
      if (part->pos[0] < cell->loc[0] ||
          part->pos[0] >= cell->loc[0] + cell->width[0] ||
          part->pos[1] < cell->loc[1] ||
          part->pos[1] >= cell->loc[1] + cell->width[1] ||
          part->pos[2] < cell->loc[2] ||
          part->pos[2] >= cell->loc[2] + cell->width[2])
        error("Particle not in correct cell");
    }
  }
#endif
  toc("Assigning particles to cells");
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

    // If we got here we have a useful cell, so we need to label the neighbours
    // as useful too
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
