// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef GRID_POINT_HPP
#define GRID_POINT_HPP

// Standard includes
#include <memory>
#include <unordered_map>
#include <vector>

// Local includes
#include "cell.hpp"
#include "params.hpp"
#include "particle.hpp"
#include "simulation.hpp"

class GridPoint {
public:
  //! The location of the grid point
  double loc[3];

  // Prototypes for member functions (defined in grid_point.cpp)
  GridPoint(double loc[3]);
  void add_particle(Particle* part, double kernel_radius);
  void add_cell(const int cell_part_count, const double cell_mass,
                double kernel_radius);
  double getOverDensity(const double kernel_radius, Simulation *sim) const;

private:
  //! The count of particles in each kernel radius
  std::unordered_map<double, double> count_map;

  //! The mass of particles in each kernel radius
  std::unordered_map<double, double> mass_map;
};

class Grid {
public:
  //! How many kernels are we using?
  int nkernels;

  //! The kernel radii
  std::vector<double> kernel_radii;

  //! The maximum kernel radius
  double max_kernel_radius;

  //! The maximum kernel radius squared
  double max_kernel_radius2;

  //! Are we using a file of grid points?
  bool grid_from_file;

  //! The number of grid points
  int n_grid_points;

  //! The number of grid points along a side (only used if we're creating grid)
  int grid_cdim;

  //! The grid points
  std::vector<GridPoint> grid_points;

  // Prototypes for member functions (defined in grid_point.cpp)
  Grid(Parameters *params);
  ~Grid();
};

// Prototypes for grid construction (used in construct_grid_points.cpp)
double nearest(const double dx, const double box_size);
Grid *createGrid(Parameters *params);
void createGridPoints(Simulation *sim, Grid *grid);
#endif // GRID_POINT_HPP
