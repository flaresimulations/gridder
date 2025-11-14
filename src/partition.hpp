// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTITION_HPP
#define PARTITION_HPP

// Local includes
#include "grid_point.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

// Partition prototypes
void partitionCells(Simulation *sim);

// Proxy prototypes
void flagProxyCells(Simulation *sim);
void exchangeProxyCells(Simulation *sim);

// Efficient particle loading (used in both serial and MPI)
std::vector<ParticleChunk> prepareToReadParts(Simulation *sim);

#ifdef WITH_MPI
// MPI-specific particle loading prototypes
void partitionChunksForReading(std::vector<ParticleChunk> &chunks, int size);
void partitionCellsByWork(Simulation *sim, Grid *grid);
void redistributeParticles(Simulation *sim, std::vector<ParticleChunk> &chunks);
void exchangeProxyCells(Simulation *sim, std::vector<ParticleChunk> &chunks);
#endif

#endif // PARTITION_HPP
