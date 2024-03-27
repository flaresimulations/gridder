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
  std::string param_file = argv[1];

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

  // Read the parameter file
  tic();
  Parameters params;
  try {
    params.parseYAMLFile(param_file);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  toc("Reading parameter file");

  params.printAllParameters();

  // Get the metadata instance
  Metadata &metadata = Metadata::getInstance();

  // Get the kernel radii
  int nkernels;
  try {
    nkernels = params.getParameterNoDefault<int>("Kernels/nkernels");
    metadata.kernel_radii.resize(nkernels);
    for (int i = 0; i < nkernels; i++) {
      std::stringstream kernel_param;
      kernel_param << "Kernels/kernel_radius_" << i + 1;
      metadata.kernel_radii[i] =
          params.getParameterNoDefault<double>(kernel_param.str());
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Get the maximum kernel radius (they may not be in order in the parameter
  // file)
  std::sort(metadata.kernel_radii.begin(), metadata.kernel_radii.end());
  metadata.max_kernel_radius = metadata.kernel_radii[nkernels - 1];

  // Get the key we should use to read the mean density
  try {
    metadata.density_key =
        params.getParameterNoDefault<std::string>("Metadata/density_key");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Get the input file path
  std::string input_file;
  try {
    input_file = params.getParameterNoDefault<std::string>("Input/filepath");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Read the rest of the metadata from the input file
  try {
    readMetadata(input_file);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Report interesting things
  message("Running with %d dark matter particles", metadata.nr_dark_matter);
  message("Mean density at z=%.2f: %e Msun / Mpc^3", metadata.redshift,
          metadata.mean_density);
  stringstream ss;
  ss << "Kernel radii (nkernels=%d):";
  for (int i = 0; i < nkernels; i++) {
    ss << " " << metadata.kernel_radii[i] << ",";
  }
  message(ss.str().c_str(), nkernels);
  message("Max kernel radius: %f", metadata.max_kernel_radius);

  // Get the cell array itself
  tic();
  std::vector<Cell *> cells;
  try {
    getTopCells(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Creating top level cells");

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
    assignPartsToGridPoints(cells);
  } catch (const std::exception &e) {
    report_error();
    return 1;
  }
  toc("Assigning particles to grid points");

  // Exit properly in MPI land
  MPI_Finalize();
  return 0;
}
