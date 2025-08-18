// Standard includes

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
  grid->grid_points.reserve(n_grid_points);

  // Create the grid points (we'll loop over every individual grid point for
  // better parallelism)
#pragma omp parallel for
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

#pragma omp critical
    {
      // Create the grid point and add it to the vector
      // TODO: We could use a tbb::concurrent_vector for grid points to
      // avoid the need for a critical section here
      grid->grid_points.emplace_back(loc);
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
  } else {
    createGridPointsEverywhere(sim, grid);
  }

  toc("Creating grid points");
}
