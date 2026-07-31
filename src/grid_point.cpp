// Standard includes
#include <algorithm>
#include <cmath>
#include <memory>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "metadata.hpp"
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
 * @brief Allocate count and mass storage for each configured kernel
 *
 * @param kernel_count The number of configured kernels
 */
void GridPoint::initializeKernels(const std::size_t kernel_count) {
  this->kernel_data.assign(kernel_count, KernelAccumulator{});
}

/**
 * @brief Add a particle to the grid point
 *
 * @param part The particle to add
 * @param kernel_index The index of the kernel to update
 */
void GridPoint::add_particle(const Particle *part,
                             const std::size_t kernel_index) {
  KernelAccumulator &kernel = this->kernel_data[kernel_index];
  kernel.count++;
  kernel.mass += part->mass;
}

/**
 * @brief Add a cell to the grid point
 *
 * @param cell_part_count The number of particles in the cell
 * @param cell_mass The mass contained in the cell
 * @param kernel_index The index of the kernel to update
 */
void GridPoint::add_cell(const std::size_t cell_part_count,
                         const double cell_mass,
                         const std::size_t kernel_index) {
  KernelAccumulator &kernel = this->kernel_data[kernel_index];
  kernel.count += cell_part_count;
  kernel.mass += cell_mass;
}

// Method to get over density inside kernel radius
double GridPoint::getOverDensity(const std::size_t kernel_index,
                                  const double kernel_radius,
                                  Simulation *sim) const {
  // Compute the volume of the kernel
  const double kernel_volume =
      (4.0 / 3.0) * M_PI * kernel_radius * kernel_radius * kernel_radius;

  // Compute the density
  const double density = getMass(kernel_index) / kernel_volume;

  // Compute the over density
  return (density / sim->mean_density) - 1;
}

// Method to get the mass inside the kernel radius
double GridPoint::getMass(const std::size_t kernel_index) const {
  return this->kernel_data[kernel_index].mass;
}

// Method to get the particle count inside the kernel radius
int GridPoint::getCount(const std::size_t kernel_index) const {
  return static_cast<int>(this->kernel_data[kernel_index].count);
}

#ifdef DEBUGGING_CHECKS
// Method to set the brute force count for a kernel
void GridPoint::setBruteForceCount(const std::size_t kernel_index,
                                   const int count) {
  this->kernel_data[kernel_index].brute_force_count = count;
}

// Method to get the brute force count for a kernel
int GridPoint::getBruteForceCount(const std::size_t kernel_index) const {
  return this->kernel_data[kernel_index].brute_force_count;
}
#endif

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

  // Determine how we are creating the grid points
  std::string grid_type =
      params->getParameter<std::string>("Grid/type", "uniform");
  if (grid_type == "file") {
    this->grid_from_file = true;
    this->grid_uniform = false;
    this->grid_random = false;
    message("Grid points will be read from file: %s",
            params->getParameter<std::string>("Grid/grid_file", "").c_str());
  } else if (grid_type == "uniform") {
    this->grid_from_file = false;
    this->grid_uniform = true;
    this->grid_random = false;
    message(
        "Grid points will be created uniformly across the simulation volume");
  } else if (grid_type == "random") {
    this->grid_from_file = false;
    this->grid_uniform = false;
    this->grid_random = true;
    message(
        "Grid points will be created randomly within the simulation volume");
  } else {
    throw std::runtime_error("Invalid grid type specified: " + grid_type);
  }

  // If we are doing a uniform grid we need the grid cdim but don't need a file
  // path or n_grid_points
  if (this->grid_uniform) {
    this->grid_cdim = params->getParameterNoDefault<int>("Grid/cdim");
    this->n_grid_points = this->grid_cdim * this->grid_cdim * this->grid_cdim;
    this->grid_file = "";
  } else if (this->grid_random) {
    // If we are doing a random grid we need the number of grid points but not
    // the cdim or file path
    this->n_grid_points =
        params->getParameterNoDefault<int>("Grid/n_grid_points");
    this->grid_cdim = 0;
    this->grid_file = "";
    // Get random seed (default to 42 for backward compatibility)
    this->random_seed = params->getParameter<int>("Grid/random_seed", 42);
    message("Using random seed: %d", this->random_seed);
  } else {
    // If we are reading from a file, get the file path and apply placeholder replacement
    std::string raw_grid_file = params->getParameter<std::string>("Grid/grid_file", "");
    Metadata *metadata = &Metadata::getInstance();
    this->grid_file = getGridFilePath(params, metadata->nsnap, raw_grid_file);
    this->grid_cdim = 0;     // Not used when reading from file
    this->n_grid_points = 0; // We'll count these when reading the file
  }
}

/**
 * @brief Destroy the Grid object
 */
Grid::~Grid() {
  // Clear the grid points
  this->grid_points.clear();
}
