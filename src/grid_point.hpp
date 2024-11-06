// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef GRID_POINT_HPP
#define GRID_POINT_HPP

// Standard includes
#include <algorithm>
#include <cmath>
#include <map>
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

  //! The count of particles in each kernel radius
  std::unordered_map<double, double> count_map;

  //! The mass of particles in each kernel radius
  std::unordered_map<double, double> mass_map;

  // Constructor
  GridPoint(double loc[3]) {
    this->loc[0] = loc[0];
    this->loc[1] = loc[1];
    this->loc[2] = loc[2];
  }

  /**
   * @brief Add a particle to the grid point
   *
   * @param part The particle to add
   * @param kernel_radius The kernel radius
   */
  void add_particle(std::shared_ptr<Particle> part, double kernel_radius) {
    this->count_map[kernel_radius]++;
    this->mass_map[kernel_radius] += part->mass;
  }

  /**
   * @brief Add a cell to the grid point
   *
   * @param cell_part_count The number of particles in the cell
   * @param cell_mass The mass contained in the cell
   * @param kernel_radius The kernel radius
   */
  void add_cell(const int cell_part_count, const double cell_mass,
                double kernel_radius) {
    this->count_map[kernel_radius] += cell_part_count;
    this->mass_map[kernel_radius] += cell_mass;
  }

  // Method to get over density inside kernel radius
  double getOverDensity(const double kernel_radius, Simulation *sim) {
    // Compute the volume of the kernel
    const double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

    // Compute the density
    const double density = this->mass_map[kernel_radius] / kernel_volume;

    // Compute the over density
    return (density / sim->mean_density) - 1;
  }
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
  std::vector<std::shared_ptr<GridPoint>> grid_points;

  /**
   * @brief Construct a new Grid object
   *
   * @param kernel_radii The kernel radii
   * @param grid_from_file Are we using a file of grid points?
   */
  Grid(Parameters *params) {

    // How many kernels will each grid point have?
    this->nkernels = params->getParameterNoDefault<int>("Kernels/nkernels");

    // Populate the kernel radii
    this->kernel_radii.resize(this->nkernels);
    for (int i = 0; i < this->nkernels; i++) {

      // Get the kernel key
      std::stringstream kernel_param;
      kernel_param << "Kernels/kernel_radius_" << i + 1;

      // Get the kernel radius
      this->kernel_radii[i] =
          params->getParameterNoDefault<double>(kernel_param.str());
    }

    // Ensure we have some kernel radii
    if (this->kernel_radii.size() == 0) {
      throw std::runtime_error("No kernel radii were provided. Ensure Kernels/"
                               "nkernels and Kernels/kernel_radius_* are set.");
    }

    // Get the maximum kernel radius
    this->max_kernel_radius =
        *std::max_element(this->kernel_radii.begin(), this->kernel_radii.end());

    // Get the maximum kernel radius squared
    this->max_kernel_radius2 =
        this->max_kernel_radius * this->max_kernel_radius;

    // Has a grid file been provided? If so Grid/grid_file will exist
    this->grid_from_file = params->exists("Grid/grid_file");

    // If we are not reading a file, get the number of grid points along an axis
    if (!this->grid_from_file) {
      this->grid_cdim = params->getParameterNoDefault<int>("Grid/cdim");
    } else {
      this->grid_cdim = 0;
    }

    // Get the number of grid points (either read from a parameter if loading
    // a file or calculated from the grid_cdim)
    if (this->grid_from_file) {
      this->n_grid_points =
          params->getParameterNoDefault<int>("Grid/n_grid_points");
    } else {
      this->n_grid_points = this->grid_cdim * this->grid_cdim * this->grid_cdim;
    }
  }

  // Destructor
  ~Grid() {
    // Clear the grid points
    this->grid_points.clear();
  }
};

// Prototypes for grid construction (used in construct_grid_points.cpp)
double nearest(const double dx, const double box_size);
Grid *createGrid(Parameters *params);
std::vector<std::shared_ptr<GridPoint>>
createGridPointsFromFile(Simulation *sim);
std::vector<std::shared_ptr<GridPoint>>
createGridPointsEverywhere(std::shared_ptr<Cell> *cells);
#endif // GRID_POINT_HPP
