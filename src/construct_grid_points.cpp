// Standard includes
#include <cmath>
#include <memory>
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

  message("Generating random grid points with parallel redistribution",
          metadata->rank);

  // Each rank generates its portion of points
  int points_per_rank = nr_grid_points / metadata->size;
  int remainder = nr_grid_points % metadata->size;

  // Distribute remainder points among first few ranks
  int my_points = points_per_rank + (metadata->rank < remainder ? 1 : 0);

  // Use same seed but different starting index for each rank to ensure
  // reproducibility
  std::mt19937 rng(42); // Fixed seed for reproducible results

  // Calculate how many random numbers to skip for this rank
  int skip_count = 0;
  for (int r = 0; r < metadata->rank; r++) {
    skip_count += (points_per_rank + (r < remainder ? 1 : 0)) * 3;
  }
  rng.discard(skip_count);

  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  // Generate my portion of random points
  std::vector<std::vector<double>> send_buffers(metadata->size);
  std::vector<int> send_counts(metadata->size, 0);

  for (int i = 0; i < my_points; i++) {
    double loc[3] = {dist_x(rng), dist_y(rng), dist_z(rng)};

    // Determine which rank should own this point based on cell ownership
    Cell *cell = getCellContainingPoint(loc);
    int target_rank = cell->rank;

    // Add to appropriate send buffer
    send_buffers[target_rank].push_back(loc[0]);
    send_buffers[target_rank].push_back(loc[1]);
    send_buffers[target_rank].push_back(loc[2]);
    send_counts[target_rank]++;
  }

  message("Generated %d points, redistributing...", metadata->rank, my_points);

  // Communicate how many points each rank will send to each other rank
  std::vector<int> recv_counts(metadata->size);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);

  // Calculate send/receive displacements and total data sizes
  std::vector<int> send_displs(metadata->size, 0);
  std::vector<int> recv_displs(metadata->size, 0);
  std::vector<int> send_data_counts(metadata->size);
  std::vector<int> recv_data_counts(metadata->size);

  for (int r = 0; r < metadata->size; r++) {
    send_data_counts[r] = send_counts[r] * 3; // 3 coordinates per point
    recv_data_counts[r] = recv_counts[r] * 3;

    if (r > 0) {
      send_displs[r] = send_displs[r - 1] + send_data_counts[r - 1];
      recv_displs[r] = recv_displs[r - 1] + recv_data_counts[r - 1];
    }
  }

  // Flatten send data
  std::vector<double> send_data;
  for (int r = 0; r < metadata->size; r++) {
    send_data.insert(send_data.end(), send_buffers[r].begin(),
                     send_buffers[r].end());
  }

  // Calculate total receive size
  int total_recv_size =
      recv_displs[metadata->size - 1] + recv_data_counts[metadata->size - 1];
  std::vector<double> recv_data(total_recv_size);

  // Perform the all-to-all exchange
  MPI_Alltoallv(send_data.data(), send_data_counts.data(), send_displs.data(),
                MPI_DOUBLE, recv_data.data(), recv_data_counts.data(),
                recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  // Create grid points from received data
  int total_local_points = 0;
  for (int r = 0; r < metadata->size; r++) {
    total_local_points += recv_counts[r];
  }

  grid->grid_points.reserve(total_local_points);

  for (int i = 0; i < total_local_points; i++) {
    double loc[3] = {recv_data[i * 3], recv_data[i * 3 + 1],
                     recv_data[i * 3 + 2]};
    grid->grid_points.emplace_back(loc);
  }

#else
  // Serial mode - generate all points directly
  message("Generating %d random grid points", nr_grid_points);

  // Use a fixed seed for reproducible results
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  grid->grid_points.reserve(nr_grid_points);

  for (int gid = 0; gid < nr_grid_points; gid++) {
    double loc[3] = {dist_x(rng), dist_y(rng), dist_z(rng)};
    grid->grid_points.emplace_back(loc);
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
