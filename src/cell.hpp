// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <H5Cpp.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <utility>
#include <vector>

// Local includes
#include "grid_point.hpp"
#include "logger.hpp"
#include "particle.hpp"
#include "serial_io.hpp"

class Cell : public std::enable_shared_from_this<Cell> {
public:
  // Cell metadata members
  double loc[3];
  double width[3];
  size_t part_count;
  double mass;
  bool is_split;

  // MPI information
  int rank;

  // Peano-hilbert index
  int64_t ph_ind;

  // Cell particle members
  // NOTE: particles are stored such that they are in child cell order
  std::vector<std::shared_ptr<Particle>> particles;

  // Grid points in cell
  std::vector<std::unique_ptr<GridPoint>> grid_points;

  // Child cells
  std::array<std::shared_ptr<Cell>, 8> children;

  // Parent cell
  std::shared_ptr<Cell> parent;

  // Store the neighbouring cells
  std::vector<std::shared_ptr<Cell>> neighbours;

  // Depth in the tree
  int depth;

  // Constructor
  Cell(const double loc[3], const double width[3],
       std::shared_ptr<Cell> parent = nullptr) {
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

    // Compute the peano-hilbert index
    this->peanoHilbertIndex();
  }

  // method to split this cell into 8 children (constructing an octree)
  void split() {

    // Get the metadata
    Metadata &metadata = Metadata::getInstance();

    // Update the max depth
    if (this->depth > metadata.max_depth)
      metadata.max_depth = this->depth;

    // Calculate the new width of the children
    double new_width[3] = {this->width[0] / 2.0, this->width[1] / 2.0,
                           this->width[2] / 2.0};

    // Check we actually need to split
    if (this->part_count < metadata.max_leaf_count) {
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

          // Create the child (here we have to do some pointer magic to get the
          // shared pointer to work)
          std::shared_ptr<Cell> child =
              std::make_shared<Cell>(new_loc, new_width, shared_from_this());

          // Set the rank of the child
          child->rank = this->rank;

          // Attach the particles to the child and count them while we're at it
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
          // NOTE: because grid points are unique pointers we need to move them
          // to the child before erasing them from the parent to avoid nullptrs
          // in the parent's grid_points vector
          auto it = this->grid_points.begin();
          while (it != this->grid_points.end()) {

            // Is the grid point in this child?
            if ((*it)->loc[0] >= new_loc[0] &&
                (*it)->loc[0] < new_loc[0] + new_width[0] &&
                (*it)->loc[1] >= new_loc[1] &&
                (*it)->loc[1] < new_loc[1] + new_width[1] &&
                (*it)->loc[2] >= new_loc[2] &&
                (*it)->loc[2] < new_loc[2] + new_width[2]) {
              // Move the unique_ptr to the child's grid_points
              child->grid_points.push_back(std::move(*it));

              // Remove the unique_ptr from the current vector
              it = this->grid_points.erase(it);
            } else {
              ++it; // Only increment if not erased
            }
          }

          // Split this child
          child->split();

          // Attach the child to this cell
          this->children[iprogeny] = child;
        }
      }
    }

    // Make sure the sum of child particle counts is the same as the parent
    size_t child_part_count = 0;
    for (int i = 0; i < 8; i++) {
      child_part_count += this->children[i]->part_count;
    }
    if (child_part_count != this->part_count)
      error("Particle count mismatch in cell %d (child_part_count = %d, "
            "this->part_count = %d)",
            this->ph_ind, child_part_count, this->part_count);
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

void getTopCells(std::vector<std::shared_ptr<Cell>> &cells) {
  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Open the HDF5 file
  HDF5Helper hdf(metadata.input_file);

  // Get the dark matter offsets and counts for each simulation cell
  std::vector<int64_t> counts;
  if (!hdf.readDataset<int64_t>(std::string("Cells/Counts/PartType1"), counts))
    error("Failed to read cell counts");

  // Loop over the cells and create them, storing the counts for domain
  // decomposition
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

    // Add the cell to the cells vector
    cells.push_back(cell);
  }

  // How many cells do we need to walk out? The number of cells we need to walk
  // out is the maximum kernel radius divided by the cell width
  const int nwalk =
      std::ceil(metadata.max_kernel_radius / metadata.width[0]) + 1;
  int nwalk_upper = nwalk;
  int nwalk_lower = nwalk;

  // If nwalk is greater than the number of cells in the simulation, we need to
  // walk out to the edge of the simulation
  if (nwalk > metadata.cdim[0] / 2) {
    nwalk_upper = metadata.cdim[0] / 2;
    nwalk_lower = metadata.cdim[0] / 2;
  }

  message("Looking for neighbours within %d cells", nwalk);

  // Loop over the cells attaching the pointers the neighbouring cells (taking
  // into account periodic boudnary conditions)
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

  hdf.close();
}

// Get the cell that contains a given point
std::shared_ptr<Cell>
getCellContainingPoint(const std::vector<std::shared_ptr<Cell>> &cells,
                       const double pos[3]) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the cell index
  int i = static_cast<int>(pos[0] / metadata.width[0]);
  int j = static_cast<int>(pos[1] / metadata.width[1]);
  int k = static_cast<int>(pos[2] / metadata.width[2]);

  // Get the cell index
  int cid = i * metadata.cdim[1] * metadata.cdim[2] + j * metadata.cdim[2] + k;

  // Return the cell
  return cells[cid];
}

void assignPartsAndPointsToCells(std::vector<std::shared_ptr<Cell>> &cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Open the HDF5 file
  HDF5Helper hdf(metadata.input_file);

  // Get the dark matter offsets and counts for each simulation cell
  std::vector<int64_t> offsets;
  std::vector<int64_t> counts;
  if (!hdf.readDataset<int64_t>(std::string("Cells/Counts/PartType1"), counts))
    error("Failed to read cell counts");
  if (!hdf.readDataset<int64_t>(std::string("Cells/OffsetsInFile/PartType1"),
                                offsets))
    error("Failed to read cell offsets");

  // Loop over cells attaching particles and grid points
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Get the cell
    std::shared_ptr<Cell> cell = cells[cid];

    // Skip if this cell isn't on this rank
    if (cell->rank != metadata.rank)
      continue;

    // Get the particle slice start and length
    const int offset = offsets[cid];
    const int count = counts[cid];

    // Get the particle data
    std::vector<double> poss;
    std::vector<double> masses;
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Coordinates"),
                                      poss, offset, count))
      error("Failed to read particle coordinates");
    if (!hdf.readDatasetSlice<double>(std::string("PartType1/Masses"), masses,
                                      offset, count))
      error("Failed to read particle masses");

    // Loop over the particle data making particles
    for (int p = 0; p < count; p++) {

      // Get the mass and position of the particle
      const double mass = masses[p];
      const double pos[3] = {poss[p * 3], poss[p * 3 + 1], poss[p * 3 + 2]};

      // Create the particle
      std::shared_ptr<Particle> part = std::make_shared<Particle>(pos, mass);

      // Add the mass to the cell
      cell->mass += mass;

      // Attach the particle to the cell
      cell->particles.push_back(part);
    }

    // Double check something hasn't gone wrong with the particle count
    if (cell->particles.size() != cell->part_count)
      error("Particle count mismatch in cell %d (particles.size = %d, "
            "cell->part_count = %d)",
            cid, cell->particles.size(), cell->part_count);
  }

  // With the particles done we can now move on to creating and assigning grid
  // points

  // Get the grid size and simulation box size
  int grid_cdim = metadata.grid_cdim;
  double *dim = metadata.dim;

  // Warn the user the spacing will be uneven if the simulation isn't cubic
  if (dim[0] != dim[1] || dim[0] != dim[2]) {
    message("Warning: The simulation box is not cubic. The grid spacing "
            "will be uneven. (dim= %f %f %f)",
            dim[0], dim[1], dim[2]);
  }

  // Compute the grid spacing
  double grid_spacing[3] = {dim[0] / grid_cdim, dim[1] / grid_cdim,
                            dim[2] / grid_cdim};

  // Create the grid points
  for (int i = 0; i < grid_cdim; i++) {
    for (int j = 0; j < grid_cdim; j++) {
      for (int k = 0; k < grid_cdim; k++) {
        double loc[3] = {(i + 0.5) * grid_spacing[0],
                         (j + 0.5) * grid_spacing[1],
                         (k + 0.5) * grid_spacing[2]};
        int index[3] = {i, j, k};

        // Get the cell this grid point is in
        std::shared_ptr<Cell> cell = getCellContainingPoint(cells, loc);

        // Skip if this grid point isn't on this rank
        if (cell->rank != metadata.rank)
          continue;

        // Create the grid point
        std::unique_ptr<GridPoint> grid_point =
            std::make_unique<GridPoint>(loc, index);

        // And attach the grid point to the cell
        cell->grid_points.push_back(std::move(grid_point));
      }
    }
  }
}

void splitCells(const std::vector<std::shared_ptr<Cell>> &cells) {
  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Loop over the cells and split them
  for (int cid = 0; cid < metadata.nr_cells; cid++) {

    // Skip cells that aren't on this rank
    if (cells[cid]->rank != metadata.rank)
      continue;

    cells[cid]->split();
  }
}

void recursivePairPartsToPoints(std::shared_ptr<Cell> cell,
                                std::shared_ptr<Cell> other) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Ensure we have grid points
  if (cell->grid_points.size() == 0)
    return;

  // Ensure the other cell has particles
  if (other->part_count == 0)
    return;

  // Calculate maximum cell distance corner to corner
  double max_cell_width = std::sqrt(cell->width[0] * cell->width[0] +
                                    cell->width[1] * cell->width[1] +
                                    cell->width[2] * cell->width[2]) +
                          std::sqrt(other->width[0] * other->width[0] +
                                    other->width[1] * other->width[1] +
                                    other->width[2] * other->width[2]);

  // Early exit if the cells are too far apart. Here we use the maximum kernel
  // plus the diagonal widths of the cells as a measure of the maximum possible
  // separation
  double dijx = cell->loc[0] - other->loc[0];
  double dijy = cell->loc[1] - other->loc[1];
  double dijz = cell->loc[2] - other->loc[2];
  double rij2 = dijx * dijx + dijy * dijy + dijz * dijz;
  if (rij2 > metadata.max_kernel_radius2 + max_cell_width * max_cell_width)
    return;

  // If the cell is split then we need to recurse
  // over the children
  if (cell->is_split) {
    for (int i = 0; i < 8; i++) {
      recursivePairPartsToPoints(cell->children[i], other);
    }
  } else if (other->is_split) {
    for (int i = 0; i < 8; i++) {
      recursivePairPartsToPoints(cell, other->children[i]);
    }
  } else {

    // Loop over the grid points in the cell
    for (int g = 0; g < cell->grid_points.size(); g++) {

      // Get the grid point
      GridPoint &grid_point = *cell->grid_points[g];

      // Loop over the particles in the other cell and assign them to the grid
      // point
      for (int p = 0; p < other->part_count; p++) {
        std::shared_ptr<Particle> part = other->particles[p];

        // Get the distance between the particle and the grid point
        double dx = part->pos[0] - grid_point.loc[0];
        double dy = part->pos[1] - grid_point.loc[1];
        double dz = part->pos[2] - grid_point.loc[2];
        double r2 = dx * dx + dy * dy + dz * dz;

        // If the particle is within the kernel radius of the grid point then
        // assign it
        if (r2 < metadata.max_kernel_radius2) {
          grid_point.add_particle(part);
        }
      }
    }
  }
}

void recursiveSelfPartsToPoints(std::shared_ptr<Cell> cell) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Ensure we have grid points
  if (cell->grid_points.size() == 0)
    return;

  // If the cell is split then we need to recurse
  // over the children but only if the cell is larger than the kernel radius
  if (cell->is_split && cell->width[0] > metadata.max_kernel_radius) {
    for (int i = 0; i < 8; i++) {
      recursiveSelfPartsToPoints(cell->children[i]);

      // And do the pair assignment
      for (int j = 0; j < 8; j++) {
        if (i == j)
          continue;
        recursivePairPartsToPoints(cell->children[i], cell->children[j]);
      }
    }
  } else {
    // Loop over the grid points in the cell
    for (int g = 0; g < cell->grid_points.size(); g++) {

      // Get the grid point
      GridPoint &grid_point = *cell->grid_points[g];

      // Loop over the particles in the cell and assign them to the grid point
      for (int p = 0; p < cell->part_count; p++) {
        std::shared_ptr<Particle> part = cell->particles[p];

        // Get the distance between the particle and the grid point
        double dx = part->pos[0] - grid_point.loc[0];
        double dy = part->pos[1] - grid_point.loc[1];
        double dz = part->pos[2] - grid_point.loc[2];
        double r2 = dx * dx + dy * dy + dz * dz;

        // If the particle is within the kernel radius of the grid point then
        // assign it
        if (r2 < metadata.max_kernel_radius2) {
          grid_point.add_particle(part);
        }
      }
    }
  }
}

void getKernelMasses(std::vector<std::shared_ptr<Cell>> cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

// Loop over the cells
#pragma omp parallel for
  for (std::shared_ptr<Cell> cell : cells) {

    // Skip cells that aren't on this rank
    if (cell->rank != metadata.rank)
      continue;

    // Recursively assign particles within a cell to the grid points within
    // the cell
    recursiveSelfPartsToPoints(cell);

    // Recursively assign particles within any neighbours to the grid points
    // within a cell
    for (std::shared_ptr<Cell> neighbour : cell->neighbours) {
      recursivePairPartsToPoints(cell, neighbour);
    }
  }
}

void writeGridFile(std::vector<std::shared_ptr<Cell>> cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata.output_file;

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC);

  // Create the Header group and write out the metadata
  hdf5.createGroup("Header");
  hdf5.writeAttribute<int>("Header", "Grid_CDim", metadata.grid_cdim);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", metadata.cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", metadata.dim);

  // Create the Grids group
  hdf5.createGroup("Grids");

  // Loop over the kernels and write out the grids themselves
  for (double kernel_rad : metadata.kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Define the grid shape
    std::array<hsize_t, 3> grid_shape = {
        static_cast<hsize_t>(metadata.grid_cdim),
        static_cast<hsize_t>(metadata.grid_cdim),
        static_cast<hsize_t>(metadata.grid_cdim)};

    // Create the dataset for this kernels grid data
    hdf5.createDataset<double, 3>("Grids/", kernel_name, grid_shape);

    // Write out the grid data cell by cell
    for (std::shared_ptr<Cell> cell : cells) {
      // Create the output array for this cell
      std::vector<double> grid_data;

      // Populate the output array with the grid point's overdensities and
      // find the offset into the main grid
      std::array<hsize_t, 3> start = {static_cast<hsize_t>(metadata.grid_cdim),
                                      static_cast<hsize_t>(metadata.grid_cdim),
                                      static_cast<hsize_t>(metadata.grid_cdim)};
      for (const std::unique_ptr<GridPoint> &gp : cell->grid_points) {
        grid_data.push_back(gp->getOverDensity(kernel_rad));

        if (gp->index[0] < start[0])
          start[0] = gp->index[0];
        if (gp->index[1] < start[1])
          start[1] = gp->index[1];
        if (gp->index[2] < start[2])
          start[2] = gp->index[2];
      }

      // Get the number of grid points along an axis in this slice (the cube
      // root of the number of grid points in the cell)
      int sub_grid_cdim = std::cbrt(cell->grid_points.size());
      std::array<hsize_t, 3> sub_grid_shape = {
          static_cast<hsize_t>(sub_grid_cdim),
          static_cast<hsize_t>(sub_grid_cdim),
          static_cast<hsize_t>(sub_grid_cdim)};

      // Ensure we haven't somehow lost a grid point
      if (sub_grid_cdim * sub_grid_cdim * sub_grid_cdim !=
          cell->grid_points.size()) {
        error("Number of grid points in cell is not a cube. (cell->grid_points."
              "size = %d, sub_grid_cdim^3 = %d)",
              cell->grid_points.size(),
              sub_grid_cdim * sub_grid_cdim * sub_grid_cdim);
      }

      // Write this cell's grid data to the HDF5 file
      hdf5.writeDatasetSlice<double, 3>("Grids/" + kernel_name, grid_data,
                                        start, sub_grid_shape);
    } // End of cell loop
  } // End of kernel loop

  // Close the HDF5 file
  hdf5.close();
}
#endif // CELL_HPP
