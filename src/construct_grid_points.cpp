// Standard includes
#include <array>
#include <cmath>
#include <fstream>
#include <memory>
#include <new>
#include <random>
#include <sstream>
#include <string>
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

#ifdef WITH_MPI
  // Get metadata for rank checking
  Metadata *metadata = &Metadata::getInstance();
#endif

  // In MPI mode, we only create grid points for cells owned by this rank
  // to reduce memory usage. In serial mode, we create all grid points.
  // We iterate through all possible grid point indices to maintain
  // deterministic ordering, but skip creation if not owned by this rank.

  // Note: We can't reserve the exact size in MPI mode since we don't know
  // how many grid points we'll create until we check cell ownership.
  // A rough estimate would be n_grid_points / size, but we'll let the
  // vector grow dynamically to avoid over-allocation.

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
    // Find which cell contains this grid point
    Cell *cell = getCellContainingPoint(loc);

    // Safety check - should never be null for valid simulation setup
    if (cell == nullptr) {
      error("Failed to find cell containing grid point at (%f, %f, %f)",
            loc[0], loc[1], loc[2]);
    }

    // Only create grid points for cells owned by this rank
    if (cell->rank != metadata->rank) {
      continue;
    }
#endif

    // Create the grid point and add it to the vector
    try {
      grid->grid_points.emplace_back(loc);
    } catch (const std::bad_alloc &e) {
      error("Memory allocation failed while creating grid point %d "
            "(current size: %zu). System out of memory. "
            "Try reducing n_grid_points parameter. Error: %s",
            gid, grid->grid_points.size(), e.what());
    }
  }

#ifdef WITH_MPI
  message("Rank %d: Created %zu grid points (out of %d total)",
          metadata->rank, grid->grid_points.size(), n_grid_points);
#else
  message("Created %zu grid points", grid->grid_points.size());
#endif

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

  message("Generating %d random grid points locally (rank %d)", target_points,
          metadata->rank);

  // Use reproducible but rank-dependent seed
  std::mt19937 rng(grid->random_seed + metadata->rank);
  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  try {
    grid->grid_points.reserve(target_points);
  } catch (const std::bad_alloc &e) {
    error("Memory allocation failed while reserving space for %d random grid "
          "points. "
          "Try reducing n_grid_points parameter. Error: %s",
          target_points, e.what());
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
      } catch (const std::bad_alloc &e) {
        error("Memory allocation failed while creating random grid point %d. "
              "System out of memory. Try reducing n_grid_points parameter. "
              "Error: %s",
              generated, e.what());
      }
    }
    // If not our cell, discard and try again
  }

#else
  // Serial mode - generate all points directly
  message("Generating %d random grid points", nr_grid_points);

  // Use seed from parameters for reproducible results
  std::mt19937 rng(grid->random_seed);
  std::uniform_real_distribution<double> dist_x(0.0, dim[0]);
  std::uniform_real_distribution<double> dist_y(0.0, dim[1]);
  std::uniform_real_distribution<double> dist_z(0.0, dim[2]);

  try {
    grid->grid_points.reserve(nr_grid_points);
  } catch (const std::bad_alloc &e) {
    error("Memory allocation failed while reserving space for %d random grid "
          "points. "
          "Try reducing n_grid_points parameter. Error: %s",
          nr_grid_points, e.what());
  }

  for (int gid = 0; gid < nr_grid_points; gid++) {
    double loc[3] = {dist_x(rng), dist_y(rng), dist_z(rng)};
    try {
      grid->grid_points.emplace_back(loc);
    } catch (const std::bad_alloc &e) {
      error("Memory allocation failed while creating random grid point %d. "
            "System out of memory. Try reducing n_grid_points parameter. "
            "Error: %s",
            gid, e.what());
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
 * @brief Read grid point coordinates from a text file
 *
 * Reads coordinates from a text file, ignoring lines that start with '#' or are
 * empty. Each valid line should contain three space/tab-separated coordinates:
 * x y z
 *
 * @param filename The path to the text file containing coordinates
 * @param coordinates Vector to store the read coordinates
 * @return The number of coordinates successfully read
 */
static int
readGridPointCoordinates(const std::string &filename,
                         std::vector<std::array<double, 3>> &coordinates) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open grid points file: " + filename);
  }

  coordinates.clear();
  std::string line;
  int line_number = 0;
  int valid_points = 0;

  while (std::getline(file, line)) {
    line_number++;

    // Find first non-whitespace character
    size_t first_char = line.find_first_not_of(" \t\r\n");

    // Skip empty lines and whitespace-only lines
    if (first_char == std::string::npos) {
      continue;
    }

    // Skip comments (lines starting with #)
    if (line[first_char] == '#') {
      continue;
    }

    // Parse the line for three coordinates
    std::istringstream iss(line);
    double x, y, z;

    if (iss >> x >> y >> z) {
      coordinates.push_back({x, y, z});
      valid_points++;
    } else {
      message("Warning: Skipping invalid line %d in %s: '%s'", line_number,
              filename.c_str(), line.c_str());
    }
  }

  file.close();

  if (valid_points == 0) {
    message("WARNING: No valid grid point coordinates found in file: %s",
            filename.c_str());
    message("The file is either empty or contains only comments/invalid lines.");
    message("Exiting gracefully - no grid points to process.");
    throw std::runtime_error("Empty grid file - no grid points to process");
  }

  message("Read %d valid grid point coordinates from %s", valid_points,
          filename.c_str());
  return valid_points;
}

/**
 * @brief Create grid points from a file
 *
 * @param sim The simulation object
 * @param grid The grid object
 */
static void createGridPointsFromFile(Simulation *sim, Grid *grid) {

  // Validate that grid_file parameter is set
  if (grid->grid_file.empty()) {
    throw std::runtime_error(
        "Grid:grid_file parameter not specified in parameter file");
  }

  // Read coordinates from the specified file
  std::vector<std::array<double, 3>> coordinates;
  int num_points;
  try {
    num_points = readGridPointCoordinates(grid->grid_file, coordinates);
  } catch (const std::runtime_error &e) {
    throw std::runtime_error("Failed to read grid points from file '" +
                             grid->grid_file + "': " + e.what());
  }

  // Validate the number of points matches user specification (if provided)
  if (grid->n_grid_points > 0 && grid->n_grid_points != num_points) {
    throw std::runtime_error(
        "Grid point count mismatch: parameter file specified " +
        std::to_string(grid->n_grid_points) + " points, but file '" +
        grid->grid_file + "' contains " + std::to_string(num_points) +
        " points");
  }

  // Update the grid with the actual number of points read
  grid->n_grid_points = num_points;

  // Get simulation box dimensions for validation
  double *dim = sim->dim;

  // Reserve space for grid points
  try {
    grid->grid_points.reserve(num_points);
  } catch (const std::bad_alloc &e) {
    error("Memory allocation failed while reserving space for %d grid points "
          "from file. "
          "Estimated memory needed: %.2f GB. Error: %s",
          num_points,
          (num_points * sizeof(GridPoint)) / (1024.0 * 1024.0 * 1024.0),
          e.what());
  }

  int valid_points = 0;
  int outside_box = 0;

  // Create grid points from the coordinates
  for (const auto &coord : coordinates) {
    double loc[3] = {coord[0], coord[1], coord[2]};

    // Validate that the point is within the simulation box
    if (loc[0] < 0 || loc[0] >= dim[0] || loc[1] < 0 || loc[1] >= dim[1] ||
        loc[2] < 0 || loc[2] >= dim[2]) {
      outside_box++;
      message(
          "Warning: Grid point (%.3f, %.3f, %.3f) is outside simulation box "
          "[0, %.3f] x [0, %.3f] x [0, %.3f] - skipping",
          loc[0], loc[1], loc[2], dim[0], dim[1], dim[2]);
      continue;
    }

#ifdef WITH_MPI
    // Skip grid points that are in cells we don't own
    Cell *cell = getCellContainingPoint(loc);
    if (cell && cell->rank != Metadata::getInstance().rank) {
      continue;
    }
#endif

    // Create the grid point and add it to the vector
    try {
      grid->grid_points.emplace_back(loc);
      valid_points++;
    } catch (const std::bad_alloc &e) {
      error("Memory allocation failed while creating grid point %d from file. "
            "System out of memory. Error: %s",
            valid_points + 1, e.what());
    }
  }

  if (outside_box > 0) {
    message("Warning: %d grid points were outside the simulation box and were "
            "skipped",
            outside_box);
  }

  message("Created %d grid points from file %s", valid_points,
          grid->grid_file.c_str());

  // Initialize mass and count maps for all grid points
  message("Initializing grid point maps for %d kernel radii", grid->nkernels);
  for (GridPoint &gp : grid->grid_points) {
    gp.initializeMaps(grid->kernel_radii);
  }
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
