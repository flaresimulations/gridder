// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTITION_HPP
#define PARTITION_HPP

// Local includes
#include "grid_point.hpp"
#include "simulation.hpp"

// Partition prototypes
void partitionCells(Simulation *sim, Grid *grid);

// Proxy prototypes
void flagProxyCells(Simulation *sim, Grid *grid);
void exchangeProxyCells(Simulation *sim);

#endif // PARTITION_HPP
