// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTITION_HPP
#define PARTITION_HPP

// Local includes
#include "grid_point.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

// Efficient particle loading (used in both serial and MPI)
// Returns empty vector if >75% cells useful (use full read)
// Returns chunks if <25% cells useful (use chunked read with gap filling)
std::vector<ParticleChunk> prepareToReadParts(Simulation *sim);

#ifdef WITH_MPI
// MPI partition and proxy prototypes
void partitionCells(Simulation *sim);
void flagProxyCells(Simulation *sim);
void exchangeProxyCells(Simulation *sim);
#endif

#endif // PARTITION_HPP
