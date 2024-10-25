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
#include <mpi.h>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "params.hpp"
#include "partition.hpp"
#include "serial_io.hpp"
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
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    say_hello();
  }

  // Set the MPI rank of the logger
  Logging::getInstance()->setRank(rank);
  Metadata::getInstance().rank = rank;

  // Set the snapshot number
  Metadata::getInstance().nsnap = nsnap;

  // Start the timer for the whole shebang
  start();

  if (rank == 0) {
    message("Running on %d MPI ranks", size);
  }

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

  // Get the metadata instance
  Metadata &metadata = Metadata::getInstance();

  // Read the rest of the metadata from the input file
  tic();
  try {
    readMetadata(metadata.input_file);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }
  toc("Reading metadata (including mean density calculation)");

  // Get the cell array itself
  tic();
  std::array<std::shared_ptr<Cell>> cells;
  try {
    getTopCells(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Creating top level cells");

  message("Number of top level cells: %d", metadata.nr_cells);

  // Decomose the cells over the MPI ranks (if we need to)
  if (size > 1) {
    tic();
    try {
      decomposeCells(cells);
    } catch (const std::exception &e) {
      report_error();
      return 1;
    }
    toc("Decomposing cells");
  } else {
    tic();
    // Set all cells to rank 0
    for (int cid = 0; cid < metadata.nr_cells; cid++) {
      cells[cid]->rank = 0;
    }
    toc("Setting all cells to rank 0");
  }

  // TODO: Need to create and communicate proxies

  // Now we know which cells are where we can make the grid points, and assign
  // them and the particles to the cells
  tic();
  try {
    assignPartsAndPointsToCells(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Assigning particles and grid points to cells");

  // And before we can actually get going we need to split the cells
  tic();
  try {
    splitCells(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Splitting cells");
  message("Maximum depth in the tree: %d", metadata.max_depth);

  // Now we can start the actual work... Associate particles with the grid
  // points within the maximum kernel radius
  tic();
  try {
    getKernelMasses(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Computing kernel masses");

  // We're done write the output
  tic();
  try {
    writeGridFile(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Writing output");

  // Stop the timer for the whole shebang
  finish();

  // Exit properly in MPI land
  MPI_Finalize();
  return 0;
}
