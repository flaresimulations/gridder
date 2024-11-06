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
#include "logger.hpp"
#include "params.hpp"
#include "particle.hpp"

// This is a Singleton class to store the necessary metadata used in the
// zoom_region_selection library.
class Metadata {
public:
  // Method to get the instance of the Metadata class
  static Metadata &getInstance() {
    static Metadata instance; // Create static instance of Metadata
    return instance;
  }

  // MPI information (set in main.parseCmdArgs)
  int rank;
  int size;

  // Parameter file path (set in main.parseCmdArgs)
  std::string param_file;

  // HDF5 file paths
  std::string input_file;
  std::string output_file;

  // The snapshot number we are working with (set in main.parseCmdArgs)
  int nsnap;

  // Tree properties
  int max_leaf_count;

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

void readMetadata(Parameters *params) {

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Get the maximum leaf count
  metadata->max_leaf_count =
      params->getParameter<int>("Tree/max_leaf_count", 200);

  // Get the input file path
  metadata->input_file = getInputFilePath(params, metadata->nsnap);

  // Get the output file path
  metadata->output_file = getOutputFilePath(params, metadata->nsnap);

  message("Reading data from: %s", metadata->input_file.c_str());
}

#endif // METADATA_HPP
