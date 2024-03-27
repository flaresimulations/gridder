// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTICLE_HPP
#define PARTICLE_HPP

class Particle {
public:
  // Particle metadata members
  double pos[3];
  double mass;

  // Constructor
  Particle(double pos[3], double mass) {
    this->pos[0] = pos[0];
    this->pos[1] = pos[1];
    this->pos[2] = pos[2];
    this->mass = mass;
  }
};

#endif // PARTICLE_HPP
