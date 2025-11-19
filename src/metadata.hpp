// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef METADATA_HPP
#define METADATA_HPP

// Standard includes
#include <array>
#include <string>
#include <vector>

// Local includes
#include "params.hpp"

// Forward declaration
class Simulation;
class Grid;

/**
 * @brief Structure to represent a contiguous chunk of particles for efficient
 * I/O
 *
 * When loading particles, we identify contiguous ranges of useful cells and
 * read their particles in chunks to minimize I/O calls. This struct tracks
 * metadata and temporary storage for one such chunk.
 *
 * Used in both serial and MPI builds for efficient sparse grid handling.
 */
struct ParticleChunk {
  size_t start_cell_id = 0;      ///< First cell ID in this chunk
  size_t end_cell_id = 0;        ///< Last cell ID in this chunk
  size_t start_particle_idx = 0; ///< Starting index in HDF5 particle arrays
  size_t particle_count = 0;     ///< Total number of particles in this chunk
  size_t grid_point_count = 0;   ///< Total number of grid points in this chunk
  int reading_rank = 0; ///< MPI rank assigned to read this chunk (0 in serial)

  // Temporary storage after reading (cleared after use)
  std::vector<double> masses;                   ///< Particle masses
  std::vector<std::array<double, 3>> positions; ///< Particle positions
};

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

  // Verbosity level (0=minimal, 1=rank 0 only, 2=all ranks)
  int verbosity = 1;

  // The snapshot number we are working with (set in main.parseCmdArgs)
  int nsnap;

  // Tree properties
  size_t max_leaf_count;

  //! Pointer to the simulation object (set after Simulation instantiation)
  Simulation *sim;

  //! Pointer to the grid object (set after Grid instantiation)
  Grid *grid;

  //! The fraction of Npart above below which we will fill gaps when reading
  // chunks of particles to perform fewer reads (higher, more unused particles
  // read but fewer I/O calls)
  double gap_fill_fraction;

#ifdef WITH_MPI
  //! How many cells do we have locally?
  int nr_local_cells;

  //! How many particles do we have locally?
  int nr_local_particles;

  //! Index of first particle on this rank (legacy - used for old partitioning)
  int first_local_part_ind = -1;

  //! Particle chunks for two-phase I/O optimization
  std::vector<ParticleChunk> particle_chunks;

  //! Total work assigned to this rank (npart * ngrid * nkernels)
  size_t local_work_cost = 0;
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
