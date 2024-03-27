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

  // HDF5 file paths
  std::string input_file;
  std::string output_file;

  // Some parameter strings that tell us what keys to use
  std::string density_key;

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

  // Grid properties
  int grid_cdim;

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

  // Read the internal units to cgs conversion factors
  double grams_to_internal, length_to_internal;
  hdf.readAttribute<double>(std::string("Units"),
                            std::string("Unit mass in cgs (U_M)"),
                            grams_to_internal);
  hdf.readAttribute<double>(std::string("Units"),
                            std::string("Unit length in cgs (U_L)"),
                            length_to_internal);

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
  hdf.readAttribute<double>(std::string("Cosmology"), metadata->density_key,
                            metadata->mean_density);

  // Convert mean density from internal units to Mpc and Msun
  double internal_to_Msun = 1.989e33 / grams_to_internal;
  double internal_to_Mpc = 3.086e24 / length_to_internal;
  metadata->mean_density =
      metadata->mean_density / internal_to_Msun * pow(internal_to_Mpc, 3);

  // Set the input file path
  metadata->input_file = input_file;

  // Count the cells
  metadata->nr_cells =
      metadata->cdim[0] * metadata->cdim[1] * metadata->cdim[2];

  // Report interesting things
  message("Running with %d dark matter particles", metadata->nr_dark_matter);
  message("Mean density at z=%.2f: %e Msun / Mpc^3", metadata->redshift,
          metadata->mean_density);
  std::stringstream ss;
  ss << "Kernel radii (nkernels=%d):";
  for (int i = 0; i < metadata->nkernels; i++) {
    ss << " " << metadata->kernel_radii[i] << ",";
  }
  message(ss.str().c_str(), metadata->nkernels);
  message("Max kernel radius: %f", metadata->max_kernel_radius);

  // Set the maximum kernel radius squared
  metadata->max_kernel_radius2 =
      metadata->max_kernel_radius * metadata->max_kernel_radius;
}

#endif // METADATA_HPP
