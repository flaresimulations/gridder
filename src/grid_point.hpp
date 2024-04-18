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

    return this->mass_map[kernel_radius];

    // // Compute the density
    // const double density = this->mass_map[kernel_radius] / kernel_volume;

    // // Compute the over density
    // const double over_density =
    //     (density / Metadata::getInstance().mean_density) - 1;

    // return over_density;
  }
};

#endif // GRID_POINT_HPP
