// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.

// Standard includes
#include <cmath>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

// MPI includes
#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "params.hpp"
#include "partition.hpp"
#include "talking.hpp"

int main(int argc, char *argv[]) {

  // Get the parameter file from the command line arguments
  if (argc > 4 || argc == 1) {
    std::cerr << "Usage: " << argv[0]
              << " <parameter_file> <nthreads> (optional <nsnap>)" << std::endl;
    return 1;
  }
  const std::string param_file = argv[1];
  const int nthreads = std::stoi(argv[2]);
  const int nsnap = (argc == 4) ? std::stoi(argv[3]) : 0;

  // Set the number of threads (this is a global setting)
  omp_set_num_threads(nthreads);

  // Set up the MPI environment
#ifdef WITH_MPI
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Set the MPI rank on the logger and metadata (the former only for
  // formatting the printed messages)
  Logging::getInstance()->setRank(rank);
  Metadata::getInstance().rank = rank;
  Metadata::getInstance().size = size;

  // Howdy
  if (rank == 0) {
    say_hello();
  }
#else
  // Set the rank to 0 (there is only one rank)
  Logging::getInstance()->setRank(0);

  // Howdy
  say_hello();
#endif

  // Get a local pointer to the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Set the snapshot number
  metadata->nsnap = nsnap;

  // Start the timer for the whole shebang
  start();

#ifdef WITH_MPI
  if (metadata->rank == 0) {
    message("Running on %d MPI ranks", metadata->size);
  }
#endif

  // Read the parameters from the parameter file
  Parameters params;
  tic();
  try {
    parseParams(params, param_file);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading parameters");

  // Read the rest of the metadata from the input file
  tic();
  try {
    readMetadata(metadata->input_file);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading metadata");

  // Get the cells array itself
  tic();

  // First allocate it onto the heap
  std::shared_ptr<Cell> *cells = new std::shared_ptr<Cell>[metadata->nr_cells];

  // Then read the top level cells from the input file
  try {
    getTopCells(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Creating top level cells");

  message("Number of top level cells: %d", metadata->nr_cells);

#ifdef WITH_MPI
  // Decomose the cells over the MPI ranks
  tic();
  try {
    decomposeCells(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Decomposing cells");
#endif

  // Now we know which cells are where we can make the grid points, and assign
  // them and the particles to the cells
  tic();
  try {
    assignPartsAndPointsToCells(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Assigning particles and grid points to cells");

  // And before we can actually get going we need to split the cells
  tic();
  try {
    splitCells(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Splitting cells");
  message("Maximum depth in the tree: %d", metadata->max_depth);

  // Now we can start the actual work... Associate particles with the grid
  // points within the maximum kernel radius
  tic();
  try {
    getKernelMasses(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Computing kernel masses");

#ifdef WITH_MPI
  // We're done write the output in parallel
  tic();
  try {
    writeGridFileParallel(cells, MPI_COMM_WORLD);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Writing output (in parallel)");
#else
  // We're done write the output in serial
  tic();
  try {
    writeGridFileSerial(cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Writing output (in serial)");
#endif

  // Clean everything up, we're tidy people
  delete[] cells;

  // Stop the timer for the whole shebang
  finish();

#ifdef WITH_MPI
  // Exit properly in MPI land
  MPI_Finalize();
#endif

  return 0;
}
