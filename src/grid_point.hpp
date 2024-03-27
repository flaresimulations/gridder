// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef GRID_POINT_HPP
#define GRID_POINT_HPP

// Standard includes
#include <cmath>
#include <vector>

// Local includes
#include "metadata.hpp"
#include "particle.hpp"

class GridPoint {
public:
  // Grid point metadata members
  double loc[3];

  // Pointers to the particles associated with this grid point
  // This includes all particles within the maximum kernel radius
  // of this grid point and is in distance from the grid point order
  std::vector<Particle *> parts;

  // Vector of particle distances from the grid point
  std::vector<double> part_dists;

  // Constructor
  GridPoint(double loc[3]) {
    this->loc[0] = loc[0];
    this->loc[1] = loc[1];
    this->loc[2] = loc[2];
  }

  // Method to add a particle to the grid point in distance from the grid point
  // order
  void add_particle(Particle *part) {
    // Get the distance between the particle and the grid point
    double dist = 0.0;
    for (int i = 0; i < 3; i++) {
      dist += pow(part->pos[i] - this->loc[i], 2);
    }
    dist = sqrt(dist);

    // If the particle list is empty, append the particle and its distance
    if (this->parts.size() == 0) {
      this->parts.push_back(part);
      this->part_dists.push_back(dist);
      return;
    }

    // Otherwise, loop over the particles and insert the new particle in
    // distance from the grid point order
    for (int i = 0; i < this->parts.size(); i++) {
      if (dist < this->part_dists[i]) {
        this->parts.insert(this->parts.begin() + i, part);
        this->part_dists.insert(this->part_dists.begin() + i, sqrt(dist));
        return;
      }
    }
  }

  // Method to get over density inside kernel radius
  double get_over_density(double kernel_radius) {
    // Compute the volume of the kernel
    double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

    // Initialize the over density
    double mass = 0.0;

    // Get the mean density of the universe for converting to over density
    Metadata &metadata = Metadata::getInstance();
    double mean_density = metadata.mean_density;

    // Loop over the particles
    for (int i = 0; i < this->parts.size(); i++) {
      // Skip if the particle is outside the kernel radius and compute and
      // return the over density
      if (this->part_dists[i] > kernel_radius) {
        return ((mass / kernel_volume) / mean_density) - 1.0;
      }

      // Add the mass of the particle to the over density
      mass += this->parts[i]->mass;
    }

    // If we got here the grid point had no local particles within the kernel
    // radius
    return -1.0;
  }
};

#endif // GRID_POINT_HPP
