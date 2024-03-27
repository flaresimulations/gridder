// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <H5Cpp.h>
#include <array>
#include <cmath>
#include <malloc/_malloc.h>
#include <memory>
#include <sys/_types/_int64_t.h>
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

    // Compute the peano-hilbert index
    this->peanoHilbertIndex();
  }

  // method to split this cell into 8 children (constructing an octree)
  void split() {

    // Get the metadata
    Metadata &metadata = Metadata::getInstance();

    // Calculate the new width of the children
    double new_width[3] = {this->width[0] / 2.0, this->width[1] / 2.0,
                           this->width[2] / 2.0};

    // Check we actually need to split
    if (new_width[0] < metadata.max_kernel_radius) {
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

          // Attach the child to this cell
          this->children[iprogeny] = child;

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
            }
          }

          // Split this child
          child->split();
        }
      }
    }
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

        // Get the cell this grid point is in
        std::shared_ptr<Cell> cell = getCellContainingPoint(cells, loc);

        // Skip if this grid point isn't on this rank
        if (cell->rank != metadata.rank)
          continue;

        // Create the grid point
        std::unique_ptr<GridPoint> grid_point =
            std::make_unique<GridPoint>(loc);

        // And attach the grid point to the cell
        cell->grid_points.push_back(std::move(grid_point));
      }
    }
  }
}

#endif // CELL_HPP
