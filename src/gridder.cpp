// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.

// Standard includes
#include <iostream>

// MPI includes
#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "cell.hpp"
#include "cmd_parser.hpp"
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
 * @brief Function to handle the command line arguments using robust parser
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @param rank MPI rank for error reporting
 * @param size MPI size for validation
 * @return CommandLineArgs structure with parsed arguments
 * @throws std::runtime_error if parsing fails
 */
CommandLineArgs parseCmdArgs(int argc, char *argv[], int rank = 0,
                             int size = 1) {
  // Parse arguments with comprehensive validation
  CommandLineArgs args = CommandLineParser::parse(argc, argv, rank, size);

  // Handle help request
  if (args.help_requested) {
    if (rank == 0) { // Only print from rank 0 in MPI
      CommandLineParser::printUsage(argv[0]);
    }
    // Return args with help flag set - caller should handle exit
    return args;
  }

  // Get a metadata instance and configure it
  Metadata *metadata = &Metadata::getInstance();

  // Set parsed values on metadata
  metadata->nsnap = args.nsnap;
  metadata->param_file = args.parameter_file;

  // Set the number of threads (this is a global setting)
  omp_set_num_threads(args.nthreads);

#ifdef WITH_MPI
  // Set the MPI rank on the logger and metadata
  metadata->rank = rank;
  metadata->size = size;
  Logging::getInstance()->setRank(metadata->rank);
#else
  // Set the rank to 0 (there is only one rank)
  Logging::getInstance()->setRank(0);
#endif

  return args;
}

/**
 * @brief Main function
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return int The exit status
 */
int main(int argc, char *argv[]) {

  // Handle MPI setup if we need it
  int rank, size;
#ifdef WITH_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    int ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
      std::cerr << "MPI_Init failed!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, ierr);
    }
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
  // If we're not using MPI, set the rank and size to 0
  rank = 0;
  size = 1;
#endif

  // Parse the command line arguments with robust validation
  CommandLineArgs args;
  try {
    args = parseCmdArgs(argc, argv, rank, size);

    // Handle help request
    if (args.help_requested) {
      return 0; // Exit cleanly after showing help
    }
  } catch (const std::exception &e) {
    if (rank == 0) { // Only print from rank 0 in MPI
      CommandLineParser::printError(e.what(), argv[0]);
    }
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  // Howdy
  say_hello();

  // Start the timer for the whole shebang
  start();

  // Get a local pointer to the metadata and unpack local variables we'll need
  Metadata *metadata = &Metadata::getInstance();
  std::string param_file = metadata->param_file;
#ifdef WITH_MPI
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
    if (rank == 0) {
      error("Failed to parse parameter file '%s': %s", param_file.c_str(),
            e.what());
    }
#ifdef WITH_MPI
    MPI_Finalize();
#endif
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
    metadata->sim = sim;
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
    metadata->grid = grid;
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
    partitionCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Decomposing cells");
#endif

  // Create the grid points (either from a file or tesselating the volume)
  tic();
  try {
    createGridPoints(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Creating grid points");

  // Assign the grid points to the cells
  tic();
  try {
    assignGridPointsToCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Assigning grid points to cells");

  // Now that we know what grid points go where flag which cells we care about
  // (i.e. those that contain grid points or are neighbours of cells that
  // contain grid points)
  tic();
  try {
    limitToUsefulCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Flagging useful cells");

#ifdef WITH_MPI
  // Find and flag the proxy cells at the edges of the partition
  tic();
  try {
    flagProxyCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Flagging proxy cells");
#endif

  // Now we know which cells are where we can make the grid points, and assign
  // them and the particles to the cells
  tic();
  try {
    assignPartsToCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Assigning particles to cells");

  // Communicate the proxy cell particles
#ifdef WITH_MPI
  tic();
  try {
    exchangeProxyCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Exchanging proxy cells");
#endif

  // And before we can actually get going we need to split the cells into the
  // cell tree. Each top level cell will become the root of an octree that
  // we can walk as we search for particles to associate with grid points
  message("Splitting cells into octrees...");
  tic();
  try {
    splitCells(sim);
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
    getKernelMasses(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Computing kernel masses");

#ifdef WITH_MPI
  // We're done write the output in parallel
  tic();
  try {
    writeGridFileParallel(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Writing output (in parallel)");
#else
  // We're done write the output in serial
  tic();
  try {
    writeGridFileSerial(sim, grid);
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
