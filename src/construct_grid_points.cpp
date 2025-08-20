// Standard includes
#include <cmath>
#include <memory>
#include <new>
#include <random>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

/**
 * @brief Find the smallest distance dx along one axis within a box of size
 * box_size
 *
 * This macro evaluates its arguments exactly once.
 *
 * Only wraps once. If dx > 2b, the returned value will be larger than b.
 * Similarly for dx < -b.
 *
 */
double nearest(const double dx, const double box_size) {

  return ((dx > 0.5 * box_size)
              ? (dx - box_size)
              : ((dx < -0.5 * box_size) ? (dx + box_size) : dx));
}

/**
 * @brief Create the grid itself
 *
 * This will create the Grid object and populate it with the grid's properties
 * but grid points will be made later after we have the partition.
 *
 * @param params The parameters for the simulation
 */
Grid *createGrid(Parameters *params) {

  tic();

  // Create the grid object
  Grid *grid = new Grid(params);

  toc("Creating grid object");

  // Return the grid
  return grid;
}

/**
 * @brief Create grid points spanning the whole simulation volume
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
static void createGridPointsEverywhere(Simulation *sim, Grid *grid) {
  // Get the grid size and simulation box size
  int grid_cdim = grid->grid_cdim;
  double *dim = sim->dim;
  int n_grid_points = grid->n_grid_points;

  // Warn the user the spacing will be uneven if the simulation isn't cubic
  if (dim[0] != dim[1] || dim[0] != dim[2]) {
    message("Warning: The simulation box is not cubic. The grid spacing "
            "will be uneven. (dim= %f %f %f)",
            dim[0], dim[1], dim[2]);
  }

  // Compute the grid spacing
  double grid_spacing[3] = {dim[0] / grid_cdim, dim[1] / grid_cdim,
                            dim[2] / grid_cdim};

  message("Have a grid spacing of %f %f %f", grid_spacing[0], grid_spacing[1],
          grid_spacing[2]);

  // Reserve space for grid points to avoid reallocations
  try {
    grid->grid_points.reserve(n_grid_points);
  } catch (const std::bad_alloc& e) {
    error("Memory allocation failed while reserving space for %d grid points. "
          "Try reducing n_grid_points parameter (current: %d). "
          "Estimated memory needed: %.2f GB. Error: %s", 
          n_grid_points, n_grid_points, 
          (n_grid_points * sizeof(GridPoint)) / (1024.0 * 1024.0 * 1024.0), 
          e.what());
  }

  // Create the grid points
  for (int gid = 0; gid < n_grid_points; gid++) {

    // Convert the flat index to the ijk coordinates of the grid point
    int i = gid / (grid_cdim * grid_cdim);
    int j = (gid / grid_cdim) % grid_cdim;
    int k = gid % grid_cdim;

    // NOTE: Important to see here we are adding 0.5 to the grid point so
    // the grid points start at 0.5 * grid_spacing and end at
    // (grid_cdim - 0.5) * grid_spacing
    double loc[3] = {(i + 0.5) * grid_spacing[0], (j + 0.5) * grid_spacing[1],
                     (k + 0.5) * grid_spacing[2]};

#ifdef WITH_MPI
    // Skip grid points that are in cells we don't own
    Cell *cell = getCellContainingPoint(loc);
    if (cell->rank != Metadata::getInstance().rank) {
      continue;
    }
#endif

    // Create the grid point and add it to the vector
    try {
      grid->grid_points.emplace_back(loc);
    } catch (const std::bad_alloc& e) {
      error("Memory allocation failed while creating grid point %d "
            "(current size: %zu). System out of memory. "
            "Try reducing n_grid_points parameter. Error: %s", 
            gid, grid->grid_points.size(), e.what());
    }
  }

  message("Created %zu grid points", grid->grid_points.size());

  // Initialize mass and count maps for all grid points
  message("Initializing grid point maps for %d kernel radii", grid->nkernels);
  for (GridPoint &gp : grid->grid_points) {
    gp.initializeMaps(grid->kernel_radii);
  }
}

/**
 * @brief Create randomly distributed grid points.
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
static void createGridPointsRandom(Simulation *sim, Grid *grid) {
  // Get the grid size and simulation box size
  int nr_grid_points = grid->n_grid_points;
  double *dim = sim->dim;

#ifdef WITH_MPI
  Metadata *metadata = &Metadata::getInstance();

  // Each rank needs to generate nr_grid_points / size local points
  int points_per_rank = nr_grid_points / metadata->size;
  int remainder = nr_grid_points % metadata->size;
  int target_points = points_per_rank + (metadata->rank < remainder ? 1 : 0);

  message("Generating %d random grid points locally (rank %d)", target_points, metadata->rank);

  // Use reproducible but rank-dependent seed
  std::mt19937 rng(42 + metadata->rank);
  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  try {
    grid->grid_points.reserve(target_points);
  } catch (const std::bad_alloc& e) {
    error("Memory allocation failed while reserving space for %d random grid points. "
          "Try reducing n_grid_points parameter. Error: %s", target_points, e.what());
  }

  int generated = 0;
  while (generated < target_points) {
    double loc[3] = {dist_x(rng), dist_y(rng), dist_z(rng)};

    // Check if this point is in a cell we own
    Cell *cell = getCellContainingPoint(loc);
    if (cell->rank == metadata->rank) {
      try {
        grid->grid_points.emplace_back(loc);
        generated++;
      } catch (const std::bad_alloc& e) {
        error("Memory allocation failed while creating random grid point %d. "
              "System out of memory. Try reducing n_grid_points parameter. "
              "Error: %s", generated, e.what());
      }
    }
    // If not our cell, discard and try again
  }

#else
  // Serial mode - generate all points directly
  message("Generating %d random grid points", nr_grid_points);

  // Use a fixed seed for reproducible results
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  try {
    grid->grid_points.reserve(nr_grid_points);
  } catch (const std::bad_alloc& e) {
    error("Memory allocation failed while reserving space for %d random grid points. "
          "Try reducing n_grid_points parameter. Error: %s", nr_grid_points, e.what());
  }

  for (int gid = 0; gid < nr_grid_points; gid++) {
    double loc[3] = {dist_x(rng), dist_y(rng), dist_z(rng)};
    try {
      grid->grid_points.emplace_back(loc);
    } catch (const std::bad_alloc& e) {
      error("Memory allocation failed while creating random grid point %d. "
            "System out of memory. Try reducing n_grid_points parameter. "
            "Error: %s", gid, e.what());
    }
  }
#endif

  message("Created %zu random grid points", grid->grid_points.size());

  // Initialize mass and count maps for all grid points
  message("Initializing grid point maps for %d kernel radii", grid->nkernels);
  for (GridPoint &gp : grid->grid_points) {
    gp.initializeMaps(grid->kernel_radii);
  }
}

/**
 * @brief Create grid points from a file
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
static void createGridPointsFromFile(Simulation * /* sim */,
                                     Grid * /* grid */) {
  error("%s is not implemented", __func__);
}

/**
 * @brief Create the grid points
 *
 * This function will create the grid points for the simulation. This can be
 * done in two ways:
 *
 * 1. Create grid points spanning the whole simulation volume.
 * 2. Create grid points from a file of positions.
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
void createGridPoints(Simulation *sim, Grid *grid) {

  tic();

  // Call the appropriate function to create the grid points
  if (grid->grid_from_file) {
    createGridPointsFromFile(sim, grid);
  } else if (grid->grid_uniform) {
    createGridPointsEverywhere(sim, grid);
  } else if (grid->grid_random) {
    createGridPointsRandom(sim, grid);
  } else {
    error("Unknown grid type");
  }

  toc("Creating grid points");
}
