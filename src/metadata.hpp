// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef METADATA_HPP
#define METADATA_HPP

// Standard includes
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// Local includes
#include "particle.hpp"
#include "serial_io.hpp"

// This is a Singleton class to store the necessary metadata used in the
// zoom_region_selection library.
class Metadata {
public:
  // Method to get the instance of the Metadata class
  static Metadata &getInstance() {
    static Metadata instance; // Create static instance of Metadata
    return instance;
  }

  // MPI information
  int rank;
  int size;

  // HDF5 file paths
  std::string input_file;
  std::string output_file;

  // The snapshot number we are working with
  int nsnap;

  // Cosmology parameters
  double redshift;
  double mean_density;

  // Kernel information
  std::vector<double> kernel_radii;
  int nkernels;
  double max_kernel_radius;
  double max_kernel_radius2;

  // Particle properties
  size_t nr_dark_matter;

  // Cell properties
  size_t nr_cells;
  int cdim[3];
  double width[3];
  int max_depth = 0;

  // Simulation properties
  double dim[3];

  // Tree properties
  int max_leaf_count;

  // Grid properties
  int grid_cdim;
  int n_grid_points;

  // Deleted copy constructor and copy assignment to prevent duplication
  Metadata(const Metadata &) = delete;            // Copy constructor
  Metadata &operator=(const Metadata &) = delete; // Copy assignment operator

private:
  // Private constructor and destructor to ensure that only one instance of the
  // class is created
  Metadata() {}
  ~Metadata() {}

  // Deleted move constructor and move assignment to ensure singleton
  Metadata(Metadata &&) = delete;            // Move constructor
  Metadata &operator=(Metadata &&) = delete; // Move assignment operator
};

void readMetadata(std::string input_file) {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Set up the HDF5 object
  HDF5Helper hdf(input_file);

  // Read the metadata from the file
  hdf.readAttribute<double>(std::string("Header"), std::string("Redshift"),
                            metadata->redshift);
  int nr_particles[6];
  hdf.readAttribute<int[6]>(std::string("Header"), std::string("NumPart_Total"),
                            nr_particles);
  metadata->nr_dark_matter = nr_particles[1];
  hdf.readAttribute<int[3]>(std::string("Cells/Meta-data"),
                            std::string("dimension"), metadata->cdim);
  hdf.readAttribute<double[3]>(std::string("Cells/Meta-data"),
                               std::string("size"), metadata->width);
  hdf.readAttribute<double[3]>(std::string("Header"), std::string("BoxSize"),
                               metadata->dim);

  // Read the masses of the particles
  std::vector<double> masses;
  if (!hdf.readDataset<double>(std::string("PartType1/Masses"), masses))
    error("Failed to read particle masses");

  // Sum the masses to get the total mass
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass) shared(masses)
  for (size_t i = 0; i < metadata->nr_dark_matter; i++) {
    total_mass += masses[i];
  }

  // Calculate the mean density in 10^10 Msun / Mpc^3 (comoving units)
  metadata->mean_density =
      total_mass / (metadata->dim[0] * metadata->dim[1] * metadata->dim[2]);

  // Set the input file path
  metadata->input_file = input_file;

  // Count the cells
  metadata->nr_cells =
      metadata->cdim[0] * metadata->cdim[1] * metadata->cdim[2];

  // Count the grid points
  metadata->grid_cdim =
      metadata->grid_cdim[0] * metadata->grid_cdim[1] * metadata->grid_cdim[2];

  // Report interesting things
  message("Redshift: %f", metadata->redshift);
  message("Running with %d dark matter particles", metadata->nr_dark_matter);
  message("Mean comoving density: %e 10**10 Msun / cMpc^3",
          metadata->mean_density);
  std::stringstream ss;
  ss << "Kernel radii (nkernels=%d):";
  for (int i = 0; i < metadata->nkernels; i++) {
    ss << " " << metadata->kernel_radii[i] << ",";
  }
  message(ss.str().c_str(), metadata->nkernels);
  message("Max kernel radius: %f", metadata->max_kernel_radius);
  message("Running with %d cells", metadata->nr_cells);
  message("Cdim: %d %d %d", metadata->cdim[0], metadata->cdim[1],
          metadata->cdim[2]);
  message("Box size: %f %f %f", metadata->dim[0], metadata->dim[1],
          metadata->dim[2]);
  message("Cell size: %f %f %f", metadata->width[0], metadata->width[1],
          metadata->width[2]);

  // Set the maximum kernel radius squared
  metadata->max_kernel_radius2 =
      metadata->max_kernel_radius * metadata->max_kernel_radius;
}

#endif // METADATA_HPP
