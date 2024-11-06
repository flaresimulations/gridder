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
#include "hdf_io.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "params.hpp"
#include "simulation.hpp"
#include "talking.hpp"
#ifdef WITH_MPI
#include "partition.hpp"
#endif

/**
 * @brief Function to handle the command line arguments
 *
 * This function will parse the command line arguments and set the metadata
 * parameters accordingly.
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return bool True if the command line arguments are valid
 */
bool parseCmdArgs(int argc, char *argv[]) {
  // Get the parameter file from the command line arguments
  if (argc > 4 || argc == 1) {
    std::cerr << "Usage: " << argv[0]
              << " <parameter_file> <nthreads> (optional <nsnap>)" << std::endl;
    return false;
  }

  // Unpack the command line arguments
  const std::string param_file_str(argv[1]);
  const int nthreads = std::stoi(argv[2]);
  const int nsnap = (argc == 4) ? std::stoi(argv[3]) : 0;

  // Get a metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Set the snapshot number on the metadata
  metadata->nsnap = nsnap;

  // Set the parameter file on the metadata
  metadata->param_file = param_file_str;

  // Set the number of threads (this is a global setting)
  omp_set_num_threads(nthreads);

#ifdef WITH_MPI
  // Set up the MPI environment
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Set the MPI rank on the logger and metadata (the former only for
  // formatting the printed messages)
  Logging::getInstance()->setRank(rank);
  metadata->rank = rank;
  metadata->size = size;

#else
  // Set the rank to 0 (there is only one rank)
  Logging::getInstance()->setRank(0);

#endif

  return true;
}

/**
 * @brief Main function
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return int The exit status
 */
int main(int argc, char *argv[]) {

  // Parse the command line arguments
  if (!parseCmdArgs(argc, argv)) {
    return 1;
  }

  // Howdy
  say_hello();

  // Start the timer for the whole shebang
  start();

  // Get a local pointer to the metadata and unpack local variables we'll need
  Metadata *metadata = &Metadata::getInstance();
  std::string param_file = metadata->param_file;
  const int nsnap = metadata->nsnap;
#ifdef WITH_MPI
  const int rank = metadata->rank;
  const int size = metadata->size;

  if (rank == 0) {
    message("Running on %d MPI ranks", metadata->size);
  }
#endif

  // Read the parameters from the parameter file
  Parameters *params;
  tic();
  try {
    params = parseParams(param_file);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading parameters");

#ifdef DEBUGGING_CHECKS
  // Print the parameters
  params->printAllParameters();
#endif

  // Setup the metadata we need to carry around (some has already been set,
  // during command line argument parsing)
  tic();
  try {
    readMetadata(params);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading metadata");

  // Get all the simulation data (this will also allocate the cells array)
  // NOTE: the cell array is automatically freed when the sim object is
  // destroyed (leaves scope)
  Simulation *sim;
  tic();
  try {
    sim = new Simulation();
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading simulation data");

  // Get the grid object (this doesn't initialise the grid points yet, just the
  // object)
  Grid *grid;
  tic();
  try {
    grid = createGrid(params);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Creating grid object");

  // Define the top level cells (this will initialise the top level with their
  // location, geometry and particle counts)
  tic();
  try {
    getTopCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Creating top level cells");

  message("Number of top level cells: %d", sim->nr_cells);

#ifdef WITH_MPI
  // Decomose the cells over the MPI ranks
  tic();
  try {
    decomposeCells(sim->cells);
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
    assignPartsAndPointsToCells(sim->cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Assigning particles and grid points to cells");

  // And before we can actually get going we need to split the cells
  tic();
  try {
    splitCells(sim->cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Splitting cells");
  message("Maximum depth in the tree: %d", sim->max_depth);

  // Now we can start the actual work... Associate particles with the grid
  // points within the maximum kernel radius
  tic();
  try {
    getKernelMasses(sim->cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Computing kernel masses");

#ifdef WITH_MPI
  // We're done write the output in parallel
  tic();
  try {
    writeGridFileParallel(sim->cells, MPI_COMM_WORLD);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Writing output (in parallel)");
#else
  // We're done write the output in serial
  tic();
  try {
    writeGridFileSerial(sim->cells);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Writing output (in serial)");
#endif

  // Stop the timer for the whole shebang
  finish();

#ifdef WITH_MPI
  // Exit properly in MPI land
  MPI_Finalize();
#endif

  return 0;
}
