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

class GridPoint {
public:
  // Grid point metadata members
  double loc[3];
  int index[3];

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
  void add_particle(std::shared_ptr<Particle> part) {
    // Get the metadata
    Metadata &metadata = Metadata::getInstance();

    // Compute the distance from the grid point to the particle
    double dist = 0.0;
    for (int i = 0; i < 3; i++) {
      dist += pow(part->pos[i] - this->loc[i], 2);
    }
    dist = sqrt(dist);

    // Add the mass to all the kernel radii that this particle is within
    for (int i = 0; i < metadata.kernel_radii.size(); i++) {
      if (dist <= metadata.kernel_radii[i]) {
        this->mass_map[metadata.kernel_radii[i]] += part->mass;
      }
    }
  }

  // Method to get over density inside kernel radius
  double getOverDensity(const double kernel_radius) {
    // Compute the volume of the kernel
    const double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

    // Compute the density
    const double density = this->mass_map[kernel_radius] / kernel_volume;

    if (this->mass_map[kernel_radius] > 0) {
      message("Grid point: mass[%f] = %f", kernel_radius,
              this->mass_map[kernel_radius]);
    }

    // Compute the over density
    const double over_density =
        (density / Metadata::getInstance().mean_density) - 1;

    return over_density;
  }
};

#endif // GRID_POINT_HPP
