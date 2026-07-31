// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef GRID_POINT_HPP
#define GRID_POINT_HPP

// Standard includes
#include <cstddef>
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
  void initializeKernels(std::size_t kernel_count);
  void add_particle(const Particle *part, std::size_t kernel_index);
  void add_cell(std::size_t cell_part_count, double cell_mass,
                std::size_t kernel_index);
  double getOverDensity(std::size_t kernel_index, double kernel_radius,
                        Simulation *sim) const;
  double getMass(std::size_t kernel_index) const;
  int getCount(std::size_t kernel_index) const;

#ifdef DEBUGGING_CHECKS
  void setBruteForceCount(std::size_t kernel_index, int count);
  int getBruteForceCount(std::size_t kernel_index) const;
#endif

private:
  struct KernelAccumulator {
    std::size_t count = 0;
    double mass = 0.0;
#ifdef DEBUGGING_CHECKS
    int brute_force_count = -1;
#endif
  };

  //! Accumulators indexed identically to Grid::kernel_radii
  std::vector<KernelAccumulator> kernel_data;
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

  //! The path to the file of grid points
  std::string grid_file;

  //! Are we creating uniform grid points?
  bool grid_uniform;

  //! Are we creating grid points randomly?
  bool grid_random;

  //! The number of grid points
  int n_grid_points;

  //! The number of grid points along a side (only used if we're creating grid)
  int grid_cdim;

  //! Random seed for reproducible random grid point generation
  int random_seed;

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
