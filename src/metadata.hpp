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
#ifdef WITH_MPI
#include "parallel_io.hpp"
#else
#include "serial_io.hpp"
#endif

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

  // Kernel information
  std::vector<double> kernel_radii;
  int nkernels;
  double max_kernel_radius;
  double max_kernel_radius2;

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

void readMetadata() {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Count the grid points
  metadata->n_grid_points =
      metadata->grid_cdim * metadata->grid_cdim * metadata->grid_cdim;

  // Report interesting things
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
