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
#include "logger.hpp"
#include "particle.hpp"

// Forward declaration
class GridPoint;

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

  // Destructor
  ~Cell() {
    // Clear out all the vectors of pointers
    this->particles.clear();
    this->neighbours.clear();
    this->grid_points.clear();
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
    std::array<hsize_t, 1> mass_start = {static_cast<hsize_t>(offset)};
    std::array<hsize_t, 1> mass_count = {static_cast<hsize_t>(count)};
    std::vector<double> masses(count);
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Masses"), masses,
                                      mass_start, mass_count))
      error("Failed to read particle masses");
    std::array<hsize_t, 2> pos_start = {static_cast<hsize_t>(offset), 0};
    std::array<hsize_t, 2> pos_count = {static_cast<hsize_t>(count), 3};
    std::vector<double> poss(count * 3);
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Coordinates"),
                                      poss, pos_start, pos_count))
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

// Prototypes for functions defined in construct_cells.cpp
void getTopCells(Simulation *sim, Grid *grid);

#endif // CELL_HPP
