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
#include "partition.hpp"  // Now needed in both serial and MPI for chunked reading
#include "simulation.hpp"
#include "talking.hpp"

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

#ifdef DEBUGGING_CHECKS
  // Print the parameters
  params->printAllParameters();
#endif

  // Setup the metadata we need to carry around (some has already been set,
  // during command line argument parsing)
  try {
    readMetadata(params);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

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

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Get the grid object (this doesn't initialise the grid points yet, just the
  // object)
  Grid *grid;
  try {
    grid = createGrid(params);
    metadata->grid = grid;
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Define the top level cells (this will initialise the top level with their
  // location, geometry and particle counts)
  try {
    getTopCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  message("Number of top level cells: %d", sim->nr_cells);

#ifdef WITH_MPI
  // Partition the cells across MPI ranks (work-based partition)
  try {
    partitionCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Create the grid points (either from a file or tessellating the volume)
  try {
    createGridPoints(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Assign the grid points to the cells
  try {
    assignGridPointsToCells(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Now that we know what grid points go where flag which cells we care about
  // (i.e. those that contain grid points or are neighbours of cells that
  // contain grid points)
  try {
    limitToUsefulCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  // Flag proxy cells at partition boundaries
  try {
    flagProxyCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Determine optimal particle reading strategy (full vs chunked)
  std::vector<ParticleChunk> particle_chunks;
  try {
    particle_chunks = prepareToReadParts(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Read particles using the optimal strategy
  if (particle_chunks.empty()) {
    // >75% of cells are useful - use full read
    try {
      assignPartsToCells(sim);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  } else {
    // <25% of cells are useful - use chunked read
    try {
      readParticlesInChunks(sim, particle_chunks);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }

    // Clear chunk data to free memory
    particle_chunks.clear();
  }

  // Just to be safe check particles are all where they should be
  try {
    checkAndMoveParticles(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  // Exchange proxy cell particles
  try {
    exchangeProxyCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Clean up non-useful cells to free memory
  // After proxy exchange, we can safely deallocate particles from cells that
  // are neither useful nor proxies
  try {
    cleanupNonUsefulCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // And before we can actually get going we need to split the cells into the
  // cell tree. Each top level cell will become the root of an octree that
  // we can walk as we search for particles to associate with grid points
  try {
    splitCells(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Now we can start the actual work... Associate particles with the grid
  // points within the maximum kernel radius
  try {
    getKernelMasses(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Percolate grid points from leaf cells back to top-level cells for output
  // This must happen AFTER getKernelMasses (which needs the octree structure)
  // but BEFORE writing output (which iterates over top-level cells)
  try {
    percolateGridPointsToTop(sim);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

#ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  // We're done write the output in parallel
  try {
    writeGridFileParallel(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
#else
  // We're done write the output in serial
  try {
    writeGridFileSerial(sim, grid);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
#endif

  // Stop the timer for the whole shebang
  finish();

#ifdef WITH_MPI
  // Exit properly in MPI land
  MPI_Finalize();
#endif

  return 0;
}
