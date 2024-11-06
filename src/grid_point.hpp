// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef GRID_POINT_HPP
#define GRID_POINT_HPP

// Standard includes
#include <cmath>
#include <map>
#include <memory>
#include <vector>

// Local includes
#include "cell.hpp"
#include "metadata.hpp"
#include "particle.hpp"

/**
 * @brief Find the smallest distance dx along one axis within a box of size
 * box_size
 *
 * This macro evaluates its arguments exactly once.
 *
 * Only wraps once. If dx > 2b, the returned value will be larger than b.
 * Similarly for dx < -b.
 *
 */
double nearest(const double dx, const double box_size) {

  return ((dx > 0.5 * box_size)
              ? (dx - box_size)
              : ((dx < -0.5 * box_size) ? (dx + box_size) : dx));
}

class GridPoint {
public:
  // Grid point metadata members
  double loc[3];
  int index[3];
  int count = 0;

  // Define a map to accumulate the mass of particles within each kernel
  // radius
  std::map<double, double> mass_map;

  // Constructor
  GridPoint(double loc[3], int index[3]) {
    this->loc[0] = loc[0];
    this->loc[1] = loc[1];
    this->loc[2] = loc[2];
    this->index[0] = index[0];
    this->index[1] = index[1];
    this->index[2] = index[2];

    // Zero the mass maps
    Metadata &metadata = Metadata::getInstance();
    for (int i = 0; i < metadata.kernel_radii.size(); i++) {
      this->mass_map[metadata.kernel_radii[i]] = 0.0;
    }
  }

  // Method to add a particle to the grid point
  void add_particle(std::shared_ptr<Particle> part, double kernel_radius) {
    // Count that we've added a particle
    this->count++;

    this->mass_map[kernel_radius] += part->mass;
  }

  // Method to add a whole cell to the grid point
  void add_cell(const int cell_part_count, const double cell_mass,
                double kernel_radius) {
    // Count that we've added a particle
    this->count += cell_part_count;

    this->mass_map[kernel_radius] += cell_mass;
  }

  // Method to get over density inside kernel radius
  double getOverDensity(const double kernel_radius) {
    // Compute the volume of the kernel
    const double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

    // Compute the density
    const double density = this->mass_map[kernel_radius] / kernel_volume;

    // Compute the over density
    return (density / Metadata::getInstance().mean_density) - 1;
  }
};

std::vector<std::shared_ptr<GridPoint>> createGridPointsFromFile() {}

/**
 * @brief Create grid points at every location in the simulation box
 *
 * @param cells The cells in the simulation
 * @return std::vector<std::shared_ptr<GridPoint>> The grid points
 */
std::vector<std::shared_ptr<GridPoint>>
createGridPointsEverywhere(std::shared_ptr<Cell> *cells) {
  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the grid size and simulation box size
  int grid_cdim = metadata->grid_cdim;
  double *dim = metadata->dim;
  int n_grid_points = metadata->n_grid_points;

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

  // Create a vector to store the grid points
  std::vector<std::shared_ptr<GridPoint>> grid_points;
  grid_points.reserve(n_grid_points);

  // Create the grid points (we'll loop over every individual grid point for
  // better parallelism)
#pragma omp parallel for
  for (int gid = 0; gid < n_grid_points; gid++) {

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

#ifdef WITH_MPI
    // In MPI land we need to make sure we own the cell this grid point
    // belongs in
    std::shared_ptr<Cell> cell = getCellContainingPoint(cells, loc);
    if (cell->rank != metadata->rank)
      continue;
#endif

#pragma omp critical
    {
      // Create the grid point and add it to the vector
      grid_points.push_back(std::make_shared<GridPoint>(loc, index));
    }
  }

  return grid_points;
}

#endif // GRID_POINT_HPP
