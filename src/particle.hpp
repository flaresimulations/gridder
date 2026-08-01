// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTICLE_HPP
#define PARTICLE_HPP

// Cells store stable indices rather than owning particle objects. Particle
// properties live in separate simulation-owned arrays.
#include <cstddef>

using ParticleIndex = std::size_t;

#endif // PARTICLE_HPP
