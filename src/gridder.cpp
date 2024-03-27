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
#include "logger.hpp"
#include "metadata.hpp"
#include "params.hpp"
#include "partition.hpp"
#include "serial_io.hpp"
#include "talking.hpp"

int main(int argc, char *argv[]) {

  // Get the parameter file from the command line arguments
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
    return 1;
  }
  const std::string param_file = argv[1];

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
  try {
    readMetadata(metadata.input_file);
  } catch (const std::exception &e) {
    error(e.what());
    return 1;
  }

  // Get the cell array itself
  tic();
  std::vector<std::shared_ptr<Cell>> cells;
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

  // Right, now we can finally associate particles to grid points.
  tic();
  try {
    assignPartsAndPointsToCells(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Assigning particles and grid points to cells");

  // Exit properly in MPI land
  MPI_Finalize();
  return 0;
}
