// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef METADATA_HPP
#define METADATA_HPP

// Standard includes
#include <string>

// Local includes
#include "params.hpp"

// Forward declaration
class Simulation;
class Grid;

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
  int rank = 0;
  int size = 1;

  // Parameter file path (set in main.parseCmdArgs)
  std::string param_file;

  // HDF5 file paths
  std::string input_file;
  std::string output_file;

  // Should we write out masses?
  bool write_masses = false;

  // The snapshot number we are working with (set in main.parseCmdArgs)
  int nsnap;

  // Tree properties
  size_t max_leaf_count;

  //! Pointer to the simulation object (set after Simulation instantiation)
  Simulation *sim;

  //! Pointer to the grid object (set after Grid instantiation)
  Grid *grid;

#ifdef WITH_MPI
  //! How many cells do we have locally?
  int nr_local_cells;

  //! How many particles do we have locally?
  int nr_local_particles;

  //! Index of first particle on this rank
  int first_local_part_ind = -1;
#endif

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

// Prototype for reading metadata (defined in metadata.cpp)
void readMetadata(Parameters *params);

#endif // METADATA_HPP
