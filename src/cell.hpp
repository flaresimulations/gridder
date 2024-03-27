// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <H5Cpp.h>
#include <array>
#include <cmath>
#include <memory>
#include <sys/_types/_int64_t.h>
#include <utility>
#include <vector>

// Local includes
#include "grid_point.hpp"
#include "logger.hpp"
#include "particle.hpp"
#include "serial_io.hpp"

class Cell {
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
  std::vector<Particle> particles;

  // Grid points in cell
  std::vector<std::unique_ptr<GridPoint>> grid_points;

  // Child cells
  std::array<Cell *, 8> children;

  // Parent cell
  Cell *parent;

  // Store the neighbouring cells
  std::vector<Cell *> neighbours;

  // Constructor
  Cell(double loc[3], double width[3], Cell *parent = NULL) {
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

  // Method to attach an array of particles to the cell
  void attach_particles(std::vector<Particle> &particles) {
    this->particles = particles;
  }

  // Method to split this cell into 8 children (constructing an octree)
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

          // Create the child
          Cell child(new_loc, new_width, this);

          // Set the rank of the child
          child.rank = this->rank;

          // Attach the child to this cell
          this->children[iprogeny] = &child;

          // Attach the particles to the child and count them while we're at it
          std::vector<Particle> child_particles = child.particles;
          for (int p = 0; p < this->particles.size(); p++) {
            if (this->particles[p].pos[0] >= new_loc[0] &&
                this->particles[p].pos[0] < new_loc[0] + new_width[0] &&
                this->particles[p].pos[1] >= new_loc[1] &&
                this->particles[p].pos[1] < new_loc[1] + new_width[1] &&
                this->particles[p].pos[2] >= new_loc[2] &&
                this->particles[p].pos[2] < new_loc[2] + new_width[2]) {
              child_particles.push_back(this->particles[p]);
              child.part_count++;
            }
          }

          // Split this child
          child.split();
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

void getTopCells(std::vector<Cell *> &cells) {
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
    Cell cell(loc, metadata.width);

    // Attach the particle count to the cell
    cell.part_count = counts[cid];

    // Add the cell to the cells vector
    cells.push_back(&cell);
  }

  // How many cells do we need to walk out? The number of cells we need to walk
  // out is the maximum kernel radius divided by the cell width
  int nwalk = std::ceil(metadata.max_kernel_radius / metadata.width[0]) + 1;

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
    Cell *cell = cells[cid];

    // Loop over the 26 neighbours
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

void assignPartsToGridPoints(std::vector<Cell *> &cells) { ; }

#endif // CELL_HPP
