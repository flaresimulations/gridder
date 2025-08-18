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

  // Initialize mass_map and count_map with 0.0 for all kernel radii
  // Note: This will be called during grid point creation, so we need access to
  // kernel radii We'll handle this initialization after grid points are created
}

/**
 * @brief Initialize mass_map and count_map with 0.0 for all kernel radii
 *
 * @param kernel_radii The vector of kernel radii to initialize
 */
void GridPoint::initializeMaps(const std::vector<double> &kernel_radii) {
  for (double kernel_rad : kernel_radii) {
    this->mass_map[kernel_rad] = 0.0;
    this->count_map[kernel_rad] = 0.0;
  }
}

/**
 * @brief Add a particle to the grid point
 *
 * @param part The particle to add
 * @param kernel_radius The kernel radius
 */
void GridPoint::add_particle(Particle *part, double kernel_radius) {
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
double GridPoint::getOverDensity(const double kernel_radius,
                                 Simulation *sim) const {
  // Compute the volume of the kernel
  const double kernel_volume = (4.0 / 3.0) * M_PI * pow(kernel_radius, 3);

  // Compute the density
  const double density = getMass(kernel_radius) / kernel_volume;

  // Compute the over density
  return (density / sim->mean_density) - 1;
}

// Method to get the mass inside the kernel radius
double GridPoint::getMass(const double kernel_radius) const {
  // Check if the kernel radius exists in the mass map
  auto it = this->mass_map.find(kernel_radius);
  if (it != this->mass_map.end()) {
    return it->second;
  } else {
    return 0.0; // Return 0 if the kernel radius is not found
  }
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
  } else {
    // If we are reading from a file, get the file path
    this->grid_file = params->getParameter<std::string>("Grid/grid_file", "");
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
