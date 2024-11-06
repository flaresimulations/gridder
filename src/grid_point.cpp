// Standard includes
#include <algorithm>
#include <cmath>
#include <memory>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "params.hpp"
#include "particle.hpp"
#include "simulation.hpp"

/**
 * @brief Construct a new GridPoint object
 *
 * @param loc The location of the grid point
 */
GridPoint::GridPoint(double loc[3]) {
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
void GridPoint::add_particle(std::shared_ptr<Particle> part,
                             double kernel_radius) {
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
void GridPoint::add_cell(const int cell_part_count, const double cell_mass,
                         double kernel_radius) {
  this->count_map[kernel_radius] += cell_part_count;
  this->mass_map[kernel_radius] += cell_mass;
}

// Method to get over density inside kernel radius
double GridPoint::getOverDensity(const double kernel_radius, Simulation *sim) {
  // Compute the volume of the kernel
  const double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

  // Compute the density
  const double density = this->mass_map[kernel_radius] / kernel_volume;

  // Compute the over density
  return (density / sim->mean_density) - 1;
}

/**
 * @brief Construct a new Grid object
 *
 * @param kernel_radii The kernel radii
 * @param grid_from_file Are we using a file of grid points?
 */
Grid::Grid(Parameters *params) {

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
  this->max_kernel_radius2 = this->max_kernel_radius * this->max_kernel_radius;

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

/**
 * @brief Destroy the Grid object
 */
Grid::~Grid() {
  // Clear the grid points
  this->grid_points.clear();
}
