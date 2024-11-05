// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <utility>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "grid_point.hpp"
#include "logger.hpp"
#include "particle.hpp"

// Load the appropriate I/O headers
#ifdef WITH_MPI
#include "parallel_io.hpp"
#else
#include "serial_io.hpp"
#endif

class Cell : public std::enable_shared_from_this<Cell> {
public:
  // Cell metadata members
  double loc[3];
  double width[3];
  size_t part_count;
  double mass;
  bool is_split;

  // MPI information
  int rank = 0;
  bool is_proxy = false;

  // Peano-hilbert index
  int64_t ph_ind;

  // Cell particle members
  // NOTE: particles are stored such that they are in child cell order
  std::vector<std::shared_ptr<Particle>> particles;

  // Grid points in cell
  std::vector<std::shared_ptr<GridPoint>> grid_points;

  // Child cells
  std::array<std::shared_ptr<Cell>, 8> children;

  // Parent cell
  std::shared_ptr<Cell> parent;

  // Pointer to the top level cell
  std::shared_ptr<Cell> top;

  // Store the neighbouring cells
  std::vector<std::shared_ptr<Cell>> neighbours;

  // Depth in the tree
  int depth;

  // Constructor
  Cell(const double loc[3], const double width[3],
       std::shared_ptr<Cell> parent = nullptr,
       std::shared_ptr<Cell> top = nullptr) {
    this->loc[0] = loc[0];
    this->loc[1] = loc[1];
    this->loc[2] = loc[2];
    this->width[0] = width[0];
    this->width[1] = width[1];
    this->width[2] = width[2];
    this->parent = parent;
    this->is_split = false;
    this->mass = 0.0;
    this->part_count = 0;
    this->rank = 0;
    this->depth = parent ? parent->depth + 1 : 0;
    this->top = top;

    // Compute the peano-hilbert index
    this->peanoHilbertIndex();
  }

  // Minimum separation between this cell and another
  double min_separation2(const std::shared_ptr<Cell> &other) {

    // Get the metadata
    Metadata &metadata = Metadata::getInstance();
    double *dim = metadata.dim;

    // Compute the minimum separation
    const double thisx_min = this->loc[0];
    const double thisy_min = this->loc[1];
    const double thisz_min = this->loc[2];
    const double otherx_min = other->loc[0];
    const double othery_min = other->loc[1];
    const double otherz_min = other->loc[2];

    const double thisx_max = this->loc[0] + this->width[0];
    const double thisy_max = this->loc[1] + this->width[1];
    const double thisz_max = this->loc[2] + this->width[2];
    const double otherx_max = other->loc[0] + other->width[0];
    const double othery_max = other->loc[1] + other->width[1];
    const double otherz_max = other->loc[2] + other->width[2];

    const double dx = std::min({fabs(nearest(thisx_min - otherx_min, dim[0])),
                                fabs(nearest(thisx_min - otherx_max, dim[0])),
                                fabs(nearest(thisx_max - otherx_min, dim[0])),
                                fabs(nearest(thisx_max - otherx_max, dim[0]))});

    const double dy = std::min({fabs(nearest(thisy_min - othery_min, dim[1])),
                                fabs(nearest(thisy_min - othery_max, dim[1])),
                                fabs(nearest(thisy_max - othery_min, dim[1])),
                                fabs(nearest(thisy_max - othery_max, dim[1]))});

    const double dz = std::min({fabs(nearest(thisz_min - otherz_min, dim[2])),
                                fabs(nearest(thisz_min - otherz_max, dim[2])),
                                fabs(nearest(thisz_max - otherz_min, dim[2])),
                                fabs(nearest(thisz_max - otherz_max, dim[2]))});

    return dx * dx + dy * dy + dz * dz;
  }

  /**
   * @brief Is this cell within a grid point's kernel radius?
   *
   * @param grid_point The grid point to check
   * @param kernel_rad2 The squared kernel radius
   *
   * @return True if the cell is within the kernel radius of the grid point,
   * false otherwise.
   */
  bool inKernel(std::shared_ptr<GridPoint> grid_point,
                const double kernel_rad2) {

    // Get the boxsize from the metadata
    Metadata &metadata = Metadata::getInstance();
    double *dim = metadata.dim;

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
  bool outsideKernel(std::shared_ptr<GridPoint> grid_point,
                     const double kernel_rad2) {

    // Get the boxsize from the metadata
    Metadata &metadata = Metadata::getInstance();
    double *dim = metadata.dim;

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
      for (int p = 0; p < this->part_count; p++) {
        std::shared_ptr<Particle> part = this->particles[p];
        const double p_dx = nearest(part->pos[0] - grid_point->loc[0], dim[0]);
        const double p_dy = nearest(part->pos[1] - grid_point->loc[1], dim[1]);
        const double p_dz = nearest(part->pos[2] - grid_point->loc[2], dim[2]);
        const double p_r2 = p_dx * p_dx + p_dy * p_dy + p_dz * p_dz;
        if (p_r2 <= kernel_rad2) {
          error(
              "Particle inside kernel radius but cell outside (dx=%f, dy = % f,"
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

  // method to split this cell into 8 children (constructing an octree)
  void split() {

    // Get the metadata
    Metadata &metadata = Metadata::getInstance();

    // Update the max depth
    if (this->depth > metadata.max_depth)
      metadata.max_depth = this->depth;

#ifdef DEBUGGING_CHECKS
    // Ensure all the particles within this cell are in the correct place
    for (int p = 0; p < this->part_count; p++) {
      if (this->particles[p]->pos[0] < this->loc[0] ||
          this->particles[p]->pos[0] >= this->loc[0] + this->width[0] ||
          this->particles[p]->pos[1] < this->loc[1] ||
          this->particles[p]->pos[1] >= this->loc[1] + this->width[1] ||
          this->particles[p]->pos[2] < this->loc[2] ||
          this->particles[p]->pos[2] >= this->loc[2] + this->width[2])
        error("Particle not in correct cell");
    }

    // Ensure all the grid points within this cell are in the correct place
    for (int p = 0; p < this->grid_points.size(); p++) {
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
    if (this->part_count < metadata.max_leaf_count &&
        this->grid_points.size() <= 1) {
      this->is_split = false;
      return;
    }

    // Flag that we are splitting this cell
    this->is_split = true;

    // Loop over the 8 children creating the cells and attaching the particles
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          // Define the index of the child
          int iprogeny = k + 2 * j + 4 * i;

          // Calculate the new location of the child
          double new_loc[3];
          new_loc[0] = this->loc[0] + i * new_width[0];
          new_loc[1] = this->loc[1] + j * new_width[1];
          new_loc[2] = this->loc[2] + k * new_width[2];

          // Create the child (here we have to do some pointer magic to get
          // the shared pointer to work)
          std::shared_ptr<Cell> child = std::make_shared<Cell>(
              new_loc, new_width, shared_from_this(), this->top);

          // Set the rank of the child
          child->rank = this->rank;

          // Attach the particles to the child and count them while we're at
          // it
          for (int p = 0; p < this->part_count; p++) {
            if (this->particles[p]->pos[0] >= new_loc[0] &&
                this->particles[p]->pos[0] < new_loc[0] + new_width[0] &&
                this->particles[p]->pos[1] >= new_loc[1] &&
                this->particles[p]->pos[1] < new_loc[1] + new_width[1] &&
                this->particles[p]->pos[2] >= new_loc[2] &&
                this->particles[p]->pos[2] < new_loc[2] + new_width[2]) {
              child->particles.push_back(this->particles[p]);
              child->part_count++;
              child->mass += this->particles[p]->mass;
            }
          }

          // Attach the grid points to the child
          for (int p = 0; p < this->grid_points.size(); p++) {
            if (this->grid_points[p]->loc[0] >= new_loc[0] &&
                this->grid_points[p]->loc[0] < new_loc[0] + new_width[0] &&
                this->grid_points[p]->loc[1] >= new_loc[1] &&
                this->grid_points[p]->loc[1] < new_loc[1] + new_width[1] &&
                this->grid_points[p]->loc[2] >= new_loc[2] &&
                this->grid_points[p]->loc[2] < new_loc[2] + new_width[2]) {
              child->grid_points.push_back(this->grid_points[p]);
            }
          }

          // Split this child
          child->split();

          // Attach the child to this cell
          this->children[iprogeny] = child;
        }
      }
    }

#ifdef DEBUGGING_CHECKS
    // Make sure the sum of child particle counts is the same as the parent
    size_t child_part_count = 0;
    for (int i = 0; i < 8; i++) {
      child_part_count += this->children[i]->part_count;
    }
    if (child_part_count != this->part_count)
      error("Particle count mismatch in cell %d (child_part_count = %d, "
            "this->part_count = %d)",
            this->ph_ind, child_part_count, this->part_count);

    // Make sure the sum of the child grid point counts is the same as the
    // parent
    size_t child_grid_point_count = 0;
    for (int i = 0; i < 8; i++) {
      child_grid_point_count += this->children[i]->grid_points.size();
    }
    if (child_grid_point_count != this->grid_points.size())
      error("Grid point count mismatch in cell %d (child_grid_point_count = "
            "%d, "
            "this->grid_points.size = %d)",
            this->ph_ind, child_grid_point_count, this->grid_points.size());
#endif
  }

  void peanoHilbertIndex() {
    // Get the metadata instance
    Metadata &metadata = Metadata::getInstance();
    // Compute the necessary order of the hilbert curve
    int order = std::ceil(std::log2(metadata.cdim[0]));

    this->ph_ind = xyzToHilbertIndex(
        static_cast<int>(this->loc[0] / this->width[0]),
        static_cast<int>(this->loc[1] / this->width[1]),
        static_cast<int>(this->loc[2] / this->width[2]), order);
  }

private:
  static void rot(int n, int &x, int &y, int &z, int rx, int ry, int rz) {
    if (ry == 0) {
      if (rz == 1) {
        x = n - 1 - x;
        y = n - 1 - y;
        std::swap(x, y);
      }
      if (rx == 1) {
        x = n - 1 - x;
        z = n - 1 - z;
        std::swap(x, z);
      }
    }
  }

  static unsigned int xyzToHilbertIndex(int x, int y, int z, int order) {
    unsigned int index = 0;
    int n = 1 << order; // 2^order
    int rx, ry, rz;
    for (int s = order - 1; s >= 0; s--) {
      rx = (x >> s) & 1;
      ry = (y >> s) & 1;
      rz = (z >> s) & 1;
      unsigned int d = (rx * 4) ^ (ry * 2) ^ (rz);
      index += d << (3 * s);
      rot(n, x, y, z, rx, ry, rz);
    }
    return index;
  }
};

void getTopCells(std::shared_ptr<Cell> *cells) {
  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Open the HDF5 file
  HDF5Helper hdf(metadata.input_file);

  // Get the dark matter offsets and counts for each simulation cell
  std::vector<int64_t> counts;
  if (!hdf.readDataset<int64_t>(std::string("Cells/Counts/PartType1"), counts))
    error("Failed to read cell counts");
  hdf.close();

// Loop over the cells and create them, storing the counts for domain
// decomposition
#pragma omp parallel for
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (metadata.cdim[1] * metadata.cdim[2]);
    int j = (cid / metadata.cdim[2]) % metadata.cdim[1];
    int k = cid % metadata.cdim[2];

    // Get the cell location and width
    double loc[3] = {i * metadata.width[0], j * metadata.width[1],
                     k * metadata.width[2]};

    // Create the cell
    std::shared_ptr<Cell> cell =
        std::make_shared<Cell>(loc, metadata.width, /*parent*/ nullptr);

    // Assign the particle count in this cell
    cell->part_count = counts[cid];

    // We need to set top outside the constructor
    cell->top = cell;

    // Add the cell to the cells vector
    cells[cid] = cell;
  }

  // Now the top level cells are made we can attached the pointers to
  // neighbouring cells (this simplifies boilerplate elsewhere)

  // How many cells do we need to walk out for the biggest kernel? This is
  // the maximum distance at which we will need to consider another cell
  const int nwalk =
      std::ceil(metadata.max_kernel_radius / metadata.width[0]) + 1;
  int nwalk_upper = nwalk;
  int nwalk_lower = nwalk;

  // If nwalk is greater than the number of cells in the simulation, we need
  // to walk out to the edge of the simulation
  if (nwalk > metadata.cdim[0] / 2) {
    nwalk_upper = metadata.cdim[0] / 2;
    nwalk_lower = metadata.cdim[0] / 2;
  }

  message("Looking for neighbours within %d cells", nwalk);

  // Loop over the cells attaching the pointers the neighbouring cells (taking
  // into account periodic boundary conditions)
#pragma omp parallel for
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (metadata.cdim[1] * metadata.cdim[2]);
    int j = (cid / metadata.cdim[2]) % metadata.cdim[1];
    int k = cid % metadata.cdim[2];

    // Get the cell location and width
    double loc[3] = {i * metadata.width[0], j * metadata.width[1],
                     k * metadata.width[2]};

    // Get the cell
    std::shared_ptr<Cell> cell = cells[cid];

    // Loop over the neighbours
    int nid = 0;
    for (int ii = -nwalk; ii < nwalk + 1; ii++) {
      for (int jj = -nwalk; jj < nwalk + 1; jj++) {
        for (int kk = -nwalk; kk < nwalk + 1; kk++) {

          // Skip the cell itself
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          // Get the neighbour index
          int iii = (i + ii + metadata.cdim[0]) % metadata.cdim[0];
          int jjj = (j + jj + metadata.cdim[1]) % metadata.cdim[1];
          int kkk = (k + kk + metadata.cdim[2]) % metadata.cdim[2];
          int cjd = iii * metadata.cdim[1] * metadata.cdim[2] +
                    jjj * metadata.cdim[2] + kkk;

          // Attach the neighbour to the cell
          cell->neighbours.push_back(cells[cjd]);
        }
      }
    }
  }
}

// Get the cell that contains a given point
std::shared_ptr<Cell> getCellContainingPoint(const std::shared_ptr<Cell> *cells,
                                             const double pos[3]) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the cell index
  int i = static_cast<int>(pos[0] / metadata.width[0]);
  int j = static_cast<int>(pos[1] / metadata.width[1]);
  int k = static_cast<int>(pos[2] / metadata.width[2]);

  // Get the cell index
  int cid =
      (i * metadata.cdim[1] * metadata.cdim[2]) + (j * metadata.cdim[2]) + k;

  // Return the cell
  return cells[cid];
}

void assignPartsAndPointsToCells(std::shared_ptr<Cell> *cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Open the HDF5 file
  HDF5Helper hdf(metadata.input_file);

  // Get the dark matter offsets and counts for each simulation cell
  std::vector<int64_t> offsets;
  std::vector<int64_t> counts;
  if (!hdf.readDataset<int64_t>(std::string("Cells/Counts/PartType1"),
                                counts)) {
    error("Failed to read cell counts");
  }
  if (!hdf.readDataset<int64_t>(std::string("Cells/OffsetsInFile/PartType1"),
                                offsets)) {
    error("Failed to read cell offsets");
  }

  // Loop over cells attaching particles and grid points
  size_t total_part_count = 0;
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Get the cell
    std::shared_ptr<Cell> cell = cells[cid];

#ifdef WITH_MPI
    // Skip if this cell isn't on this rank and isn't a proxy
    if (cell->rank != metadata.rank && !cell->is_proxy)
      continue;
#endif

    // Get the particle slice start and length
    const int offset = offsets[cid];
    const int count = counts[cid];
    total_part_count += count;

    // Read the masses and positions for this cell
    std::vector<double> masses(count);
    std::vector<double> poss(count * 3);
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Masses"), masses,
                                      offset, count))
      error("Failed to read particle masses");
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Coordinates"),
                                      poss, offset * 3, count * 3))
      error("Failed to read particle positions");

    // Loop over the particle data making particles
    for (int p = 0; p < count; p++) {

      // Get the mass and position of the particle
      const double mass = masses[p];
      const double pos[3] = {poss[p * 3], poss[p * 3 + 1], poss[p * 3 + 2]};

      // Create the particle
      std::shared_ptr<Particle> part;
      try {
        part = std::make_shared<Particle>(pos, mass);
      } catch (const std::exception &e) {
        error("Failed to create particle: %s", e.what());
      }

      // Add the mass to the cell
      cell->mass += mass;

      // Attach the particle to the cell
      cell->particles.push_back(part);
    }

#ifdef DEBUGGING_CHECKS
    // Double check something hasn't gone wrong with the particle count
    if (cell->particles.size() != cell->part_count)
      error("Particle count mismatch in cell %d (particles.size = %d, "
            "cell->part_count = %d)",
            cid, cell->particles.size(), cell->part_count);
#endif
  }

#ifdef DEBUGGING_CHECKS
  // Make sure we have attached all the particles
  size_t total_cell_part_count = 0;
  for (std::shared_ptr<Cell> cell : cells) {
    total_cell_part_count += cell->part_count;
  }
  if (total_part_count != total_cell_part_count) {
    error("Particle count mismatch (total_part_count = %d, "
          "total_cell_part_count = "
          "%d)",
          total_part_count, total_cell_part_count);
  }
#endif

  // Compute the total mass in the simulation from the cells
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cid = 0; cid < metadata.nr_cells; cid++) {
    total_mass += cells[cid]->mass;
  }

#ifdef WITH_MPI
  // Reduce the total mass
  double global_total_mass = 0.0;
  MPI_Allreduce(&total_mass, &global_total_mass, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  total_mass = global_total_mass;
#endif

  // Compute the mean comoving density
  metadata.mean_density =
      total_mass / (metadata.dim[0] * metadata.dim[1] * metadata.dim[2]);

  message("Mean comoving density: %e 10**10 Msun / cMpc^3",
          metadata.mean_density);

  // With the particles done we can now move on to creating and assigning grid
  // points

  // Get the grid size and simulation box size
  int grid_cdim = metadata.grid_cdim;
  double *dim = metadata.dim;
  int nr_grid_points = grid_cdim * grid_cdim * grid_cdim;

  // Warn the user the spacing will be uneven if the simulation isn't cubic
  if (dim[0] != dim[1] || dim[0] != dim[2]) {
    message("Warning: The simulation box is not cubic. The grid spacing "
            "will be uneven. (dim= %f %f %f)",
            dim[0], dim[1], dim[2]);
  }

  // Compute the grid spacing
  double grid_spacing[3] = {dim[0] / grid_cdim, dim[1] / grid_cdim,
                            dim[2] / grid_cdim};

  message("Have a grid spacing of %f %f %f", grid_spacing[0], grid_spacing[1],
          grid_spacing[2]);

  // Create the grid points (we'll loop over every individual grid point for
  // better parallelism)
#pragma omp parallel for
  for (int gid = 0; gid < nr_grid_points; gid++) {

    // Convert the flat index to the ijk coordinates of the grid point
    int i = gid / (grid_cdim * grid_cdim);
    int j = (gid / grid_cdim) % grid_cdim;
    int k = gid % grid_cdim;

    // NOTE: Important to see here we are adding 0.5 to the grid point so
    // the grid points start at 0.5 * grid_spacing and end at
    // (grid_cdim - 0.5) * grid_spacing
    double loc[3] = {(i + 0.5) * grid_spacing[0], (j + 0.5) * grid_spacing[1],
                     (k + 0.5) * grid_spacing[2]};
    int index[3] = {i, j, k};

    // Get the cell this grid point is in
    std::shared_ptr<Cell> cell = getCellContainingPoint(cells, loc);

    // Skip if this grid point isn't on this rank
    if (cell->rank != metadata.rank)
      continue;

    // Create the grid point
    std::shared_ptr<GridPoint> grid_point =
        std::make_shared<GridPoint>(loc, index);

#pragma omp critical
    {
      // And attach the grid point to the cell
      // TODO: We could use a tbb::concurrent_vector for grid points to
      // avoid the need for a critical section here
      cell->grid_points.push_back(grid_point);
    }
  }

#ifdef DEBUGGING_CHECKS
  // Check particles are in the right cells
  for (std::shared_ptr<Cell> cell : cells) {
    for (std::shared_ptr<Particle> part : cell->particles) {
      if (part->pos[0] < cell->loc[0] ||
          part->pos[0] >= cell->loc[0] + cell->width[0] ||
          part->pos[1] < cell->loc[1] ||
          part->pos[1] >= cell->loc[1] + cell->width[1] ||
          part->pos[2] < cell->loc[2] ||
          part->pos[2] >= cell->loc[2] + cell->width[2])
        error("Particle not in correct cell");
    }
  }

  // Check grid points are in the right cells
  for (std::shared_ptr<Cell> cell : cells) {
    for (std::shared_ptr<GridPoint> grid_point : cell->grid_points) {
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
}

void splitCells(const std::shared_ptr<Cell> *cells) {
  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Loop over the cells and split them
#pragma omp parallel for
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Skip cells that aren't on this rank
    if (cells[cid]->rank != metadata.rank)
      continue;

    cells[cid]->split();
  }
}

void addPartsToGridPoint(std::shared_ptr<Cell> cell,
                         std::shared_ptr<GridPoint> grid_point,
                         const double kernel_rad, const double kernel_rad2) {

  // Get the boxsize from the metadata
  Metadata &metadata = Metadata::getInstance();
  double *dim = metadata.dim;

  // Loop over the particles in the cell and assign them to the grid point
  for (int p = 0; p < cell->part_count; p++) {
    std::shared_ptr<Particle> part = cell->particles[p];

    // Get the distance between the particle and the grid point
    double dx = nearest(part->pos[0] - grid_point->loc[0], dim[0]);
    double dy = nearest(part->pos[1] - grid_point->loc[1], dim[1]);
    double dz = nearest(part->pos[2] - grid_point->loc[2], dim[2]);
    double r2 = dx * dx + dy * dy + dz * dz;

    // If the particle is within the kernel radius of the grid point then
    // assign it
    if (r2 <= kernel_rad2) {
      grid_point->add_particle(part, kernel_rad);
    }
  }
}

void recursivePairPartsToPoints(std::shared_ptr<Cell> cell,
                                std::shared_ptr<Cell> other,
                                const double kernel_rad,
                                const double kernel_rad2) {

  // Ensure we have grid points, otherwise there's nothing to add to
  if (cell->grid_points.size() == 0)
    return;

  // Ensure the other cell has particles, otherwise there's nothing to add
  if (other->part_count == 0)
    return;

  // If we have more than one grid point recurse (we can always do this since
  // the cell tree was constructed such that the leaves have only 1 grid point)
  if (cell->grid_points.size() > 1) {
    for (int i = 0; i < 8; i++) {
      recursivePairPartsToPoints(cell->children[i], other, kernel_rad,
                                 kernel_rad2);
    }
    return;
  }

  // Ensure we only have 1 grid point now we are in a leaf
  if (cell->grid_points.size() > 1) {
    error("We shouldn't be able to find a leaf with more than 1 grid point "
          "(leaf->grid_points.size()=%d",
          cell->grid_points.size());
  }

  // Get the single grid point in this leaf
  std::shared_ptr<GridPoint> grid_point = cell->grid_points[0];

  // Early exit if the cells are too far apart.
  if (other->outsideKernel(grid_point, kernel_rad2))
    return;

  // Can we just add the whole cell to the grid point?
  if (other->inKernel(grid_point, kernel_rad2)) {
    grid_point->add_cell(other->part_count, other->mass, kernel_rad);
    return;
  }

  // Get an instance of the metadata
  Metadata &metadata = Metadata::getInstance();

  // If the other cell is split then we need to recurse over the children before
  // trying to add the particles
  if (other->is_split && other->part_count > metadata.max_leaf_count) {
    for (int i = 0; i < 8; i++) {
      recursivePairPartsToPoints(cell, other->children[i], kernel_rad,
                                 kernel_rad2);
    }
    return;
  }

  // Ok, we can't just add the whole cell to the grid point, instead check
  // the particles in the other cell
  addPartsToGridPoint(other, grid_point, kernel_rad, kernel_rad2);
}

void recursiveSelfPartsToPoints(std::shared_ptr<Cell> cell,
                                const double kernel_rad,
                                const double kernel_rad2) {

  // Ensure we have grid points and particles
  if (cell->grid_points.size() == 0 || cell->part_count == 0)
    return;

  // If the cell is split then we need to recurse over the children
  if (cell->is_split && cell->grid_points.size() > 1) {
    for (int i = 0; i < 8; i++) {
      recursiveSelfPartsToPoints(cell->children[i], kernel_rad, kernel_rad2);

      // And do the pair assignment
      for (int j = 0; j < 8; j++) {
        if (i == j)
          continue;
        recursivePairPartsToPoints(cell->children[i], cell->children[j],
                                   kernel_rad, kernel_rad2);
      }
    }
  } else {

    // Ensure we only have 1 grid point now we are in a leaf
    if (cell->grid_points.size() > 1) {
      error("We shouldn't be able to find a leaf with more than 1 grid point "
            "(leaf->grid_points.size()=%d",
            cell->grid_points.size());
    }

    // Get the single grid point in this leaf
    GridPoint &grid_point = *cell->grid_points[0];

    // If the diagonal of the cell is less than the kernel radius then we can
    // just add the whole cell to the grid point since the entire cell is
    // within the kernel radius
    const double cell_diag = cell->width[0] * cell->width[0] +
                             cell->width[1] * cell->width[1] +
                             cell->width[2] * cell->width[2];
    if (cell_diag <= kernel_rad2) {
      grid_point.add_cell(cell->part_count, cell->mass, kernel_rad);
      return;
    }

    // Associate particles to the single grid point
    addPartsToGridPoint(cell, cell->grid_points[0], kernel_rad, kernel_rad2);
  }
}

void getKernelMasses(std::shared_ptr<Cell> *cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Loop over the cells
#pragma omp parallel for
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Get the cell
    std::shared_ptr<Cell> cell = cells[cid];

    // Skip cells that aren't on this rank
    if (cell->rank != metadata.rank)
      continue;

    // Loop over kernels
    for (double kernel_rad : metadata.kernel_radii) {

      // Compute squared kernel radius
      double kernel_rad2 = kernel_rad * kernel_rad;

      // Recursively assign particles within a cell to the grid points within
      // the cell
      recursiveSelfPartsToPoints(cell, kernel_rad, kernel_rad2);

      // Recursively assign particles within any neighbours to the grid points
      // within a cell
      for (std::shared_ptr<Cell> neighbour : cell->neighbours) {
        recursivePairPartsToPoints(cell, neighbour, kernel_rad, kernel_rad2);
      }
    }
  }
}

void writeGridFileSerial(std::shared_ptr<Cell> *cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata.output_file;

  message("Writing grid data to %s", filename.c_str());

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC);

  // Create the Header group and write out the metadata
  hdf5.createGroup("Header");
  hdf5.writeAttribute<int>("Header", "NGridPoint", metadata.n_grid_points);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", metadata.cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", metadata.dim);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              metadata.max_kernel_radius);
  hdf5.writeAttribute<double>("Header", "Redshift", metadata.redshift);

  // Create the Grids group
  hdf5.createGroup("Grids");

  // Loop over cells and collect how many grid points we have in each cell
  std::vector<int> grid_point_counts(metadata.nr_cells, 0);
  for (int cid = 0; cid < metadata.nr_cells; cid++) {
    grid_point_counts[cid] = cells[cid]->grid_points.size();
  }

  // Now we have the counts convert these to a start index for each cell so we
  // can use a cell look up table to find the grid points
  std::vector<int> grid_point_start(metadata.nr_cells, 0);
  for (int cid = 1; cid < metadata.nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Write out this cell lookup table
  std::array<hsize_t, 1> sim_cell_dims = {
      static_cast<hsize_t>(metadata.nr_cells)};
  hdf5.createGroup("Cells");
  hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                            sim_cell_dims);
  hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                            sim_cell_dims);

  // Create a dataset we'll write the grid positions into
  std::array<hsize_t, 2> grid_point_positions_dims = {
      static_cast<hsize_t>(metadata.n_grid_points), static_cast<hsize_t>(3)};
  hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                grid_point_positions_dims);

  // We only want to write the positions once so lets make a flag to ensure
  // we only do this once
  bool written_positions = false;

  // Loop over the kernels and write out the grids themselves
  for (double kernel_rad : metadata.kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Create the kernel group
    hdf5.createGroup("Grids/" + kernel_name);

    // Create the grid point over densities dataset
    std::array<hsize_t, 1> grid_point_overdens_dims = {
        static_cast<hsize_t>(metadata.n_grid_points)};
    hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                      "/GridPointOverDensities",
                                  grid_point_overdens_dims);

    // Write out the grid data cell by cell
    for (int cid = 0; cid < metadata.nr_cells; cid++) {
      // Get the cell
      std::shared_ptr<Cell> cell = cells[cid];

      // Skip cells empty cells
      if (cell->grid_points.size() == 0)
        continue;

      // Get the start and end indices for this cell's grid points in the
      // global grid point array
      int start = grid_point_start[cid];

      // Create the output array for this cell
      std::vector<double> cell_grid_ovdens(grid_point_counts[cid], 0.0);
      std::vector<double> cell_grid_pos(grid_point_counts[cid] * 3, 0.0);

      // Loop over grid points and populate this cell's slices
      for (const std::shared_ptr<GridPoint> &gp : cell->grid_points) {
        // Get the over density for this grid point
        cell_grid_ovdens.push_back(gp->getOverDensity(kernel_rad));

        // If we need to get the positions then do so now
        if (!written_positions) {
          cell_grid_pos.push_back(gp->loc[0]);
          cell_grid_pos.push_back(gp->loc[1]);
          cell_grid_pos.push_back(gp->loc[2]);
        }
      }

      // Write out the grid point over densities for this cell
      hdf5.writeDatasetSlice<double, 1>(
          "Grids/" + kernel_name + "/GridPointOverDensities", cell_grid_ovdens,
          {static_cast<hsize_t>(start)},
          {static_cast<hsize_t>(grid_point_counts[cid])});

      // If we haven't written the grid point positions yet then do so now
      if (!written_positions) {
        hdf5.writeDatasetSlice<double, 2>(
            "Grids/GridPointPositions", cell_grid_pos,
            {static_cast<hsize_t>(start), 0},
            {static_cast<hsize_t>(grid_point_counts[cid]), 3});
      }

    } // End of cell Loop

    // Once we've got here we know we've written the grid point positions
    written_positions = true;

  } // End of kernel loop

  // Close the HDF5 file
  hdf5.close();
}

#ifdef WITH_MPI
/**
 * @brief Writes grid data to an HDF5 file in parallel using MPI
 *
 * This function writes simulation grid data in parallel across multiple MPI
 * ranks, with each rank writing only the data it owns. The function leverages
 * HDF5 parallel I/O capabilities, using collective operations to ensure data
 * consistency and performance.
 *
 * Each rank:
 * - Aggregates its local grid point counts, which are then reduced across all
 *   ranks so each has a consistent view of the total grid.
 * - Writes data to dedicated portions of the HDF5 file, minimizing contention.
 * - Writes global metadata (attributes) only from rank 0 to avoid duplication.
 *
 * @param cells Pointer to an array of Cell objects representing the simulation
 * grid
 * @param comm MPI communicator used for parallel I/O
 */
void writeGridFileParallel(std::shared_ptr<Cell> *cells, MPI_Comm comm) {

  // Retrieve global simulation metadata
  Metadata &metadata = Metadata::getInstance();

  // Define the output filename from metadata
  const std::string filename = metadata.output_file;
  message("Writing grid data to %s", filename.c_str());

  // Initialize the HDF5 file in parallel mode
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC, comm);

  // Only rank 0 writes global metadata attributes to the file
  if (metadata.rank == 0) {

    // Create a Header group and write simulation attributes
    hdf5.createGroup("Header");
    hdf5.writeAttribute<int>("Header", "NGridPoint", metadata.n_grid_points);
    hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", metadata.cdim);
    hdf5.writeAttribute<double[3]>("Header", "BoxSize", metadata.dim);
    hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                                metadata.max_kernel_radius);
    hdf5.writeAttribute<double>("Header", "Redshift", metadata.redshift);

    // Create a Grids group to store simulation data for multiple kernels
    hdf5.createGroup("Grids");
  }

  // Initialize a vector to hold the local grid point counts for each cell on
  // this rank
  std::vector<int> rank_grid_point_counts(metadata.nr_cells, 0);
  for (int cid = 0; cid < metadata.nr_cells; cid++) {
    // Only count cells that are owned by the current rank
    if (cells[cid]->rank == metadata.rank) {
      rank_grid_point_counts[cid] = cells[cid]->grid_points.size();
    }
  }

  // Aggregate the grid point counts across all ranks using MPI_Allreduce
  // Each rank ends up with a complete view of grid_point_counts across all
  // cells
  std::vector<int> grid_point_counts(metadata.nr_cells, 0);
  MPI_Allreduce(rank_grid_point_counts.data(), grid_point_counts.data(),
                metadata.nr_cells, MPI_INT, MPI_SUM, comm);

  // Calculate starting indices for each cell in the global grid point array
  std::vector<int> grid_point_start(metadata.nr_cells, 0);
  for (int cid = 1; cid < metadata.nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Only rank 0 writes the lookup table to map cells to grid points
  if (metadata.rank == 0) {
    std::array<hsize_t, 1> cell_dims = {
        static_cast<hsize_t>(metadata.nr_cells)};
    hdf5.createGroup("Cells");
    hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                              cell_dims);
    hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                              cell_dims);

    // Define dimensions for grid point positions dataset
    std::array<hsize_t, 2> grid_point_positions_dims = {
        static_cast<hsize_t>(metadata.n_grid_points), static_cast<hsize_t>(3)};
    hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                  grid_point_positions_dims);
  }

  // Track if grid point positions have been written to avoid redundancy
  bool written_positions = false;

  // Iterate over kernel radii to write grid data for each kernel
  for (double kernel_rad : metadata.kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Rank 0 creates the group for each kernel in the Grids group
    if (metadata.rank == 0) {
      hdf5.createGroup("Grids/" + kernel_name);
    }

    // Rank 0 also defines the dataset for storing grid point over-densities
    if (metadata.rank == 0) {
      std::array<hsize_t, 1> grid_point_overdens_dims = {
          static_cast<hsize_t>(metadata.n_grid_points)};
      hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                        "/GridPointOverDensities",
                                    grid_point_overdens_dims);
    }

    // Synchronize all ranks to ensure consistent data access
    MPI_Barrier(comm);

// Process cells and write grid data in parallel using OpenMP
#pragma omp parallel for
    for (int cid = 0; cid < metadata.nr_cells; cid++) {
      // Skip cells that are not owned by this rank or are empty
      if (cells[cid]->rank != metadata.rank || cells[cid]->grid_points.empty())
        continue;

      // Get the start index and count of grid points for this cell
      int start = grid_point_start[cid];
      int count = grid_point_counts[cid];

      // Initialize data buffers for over-densities and positions
      std::vector<double> cell_grid_ovdens(count, 0.0);
      std::vector<double> cell_grid_pos(count * 3, 0.0);

      // Fill over-density and position buffers with data from grid points
      for (size_t i = 0; i < cells[cid]->grid_points.size(); ++i) {
        const std::shared_ptr<GridPoint> &gp = cells[cid]->grid_points[i];
        cell_grid_ovdens[i] = gp->getOverDensity(kernel_rad);

        // Populate position data if positions haven't been written yet
        if (!written_positions) {
          cell_grid_pos[i * 3] = gp->loc[0];
          cell_grid_pos[i * 3 + 1] = gp->loc[1];
          cell_grid_pos[i * 3 + 2] = gp->loc[2];
        }
      }

      // Write over-density data for this cell’s grid points as a slice
      hdf5.writeDatasetSlice<double, 1>(
          "Grids/" + kernel_name + "/GridPointOverDensities", cell_grid_ovdens,
          {static_cast<hsize_t>(start)}, {static_cast<hsize_t>(count)});

      // Write grid point positions if not yet written
      if (!written_positions) {
        hdf5.writeDatasetSlice<double, 2>(
            "Grids/GridPointPositions", cell_grid_pos,
            {static_cast<hsize_t>(start), 0}, {static_cast<hsize_t>(count), 3});
      }
    } // End of OpenMP parallel for loop over cells

    // Mark positions as written after the first kernel loop
    written_positions = true;

  } // End of loop over kernel radii

  // Close the HDF5 file
  hdf5.close();
}
#endif

#endif // CELL_HPP
