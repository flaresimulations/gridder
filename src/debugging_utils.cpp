// Standard includes
#include <algorithm>
#include <cmath>
#include <vector>

// Local includes
#include "debugging_utils.hpp"
#include "cell.hpp"
#include "grid_point.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

#ifdef DEBUGGING_CHECKS

/**
 * @brief Validate that grid points are correctly assigned to their containing
 * cells
 *
 * This checks that each grid point is spatially within the bounds of the cell
 * it's been assigned to.
 */
void validateGridPointCellAssignment(Simulation *sim,
                                      [[maybe_unused]] Grid *grid) {
  message("[DEBUG] Validating grid point to cell assignments...");

  int errors = 0;
  std::vector<Cell> &cells = sim->cells;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    for (GridPoint *gp : cell->grid_points) {
      // Check if grid point is within cell bounds
      bool inside_x = (gp->loc[0] >= cell->loc[0]) &&
                      (gp->loc[0] < cell->loc[0] + cell->width[0]);
      bool inside_y = (gp->loc[1] >= cell->loc[1]) &&
                      (gp->loc[1] < cell->loc[1] + cell->width[1]);
      bool inside_z = (gp->loc[2] >= cell->loc[2]) &&
                      (gp->loc[2] < cell->loc[2] + cell->width[2]);

      if (!inside_x || !inside_y || !inside_z) {
        message("[DEBUG] ERROR: Grid point at (%.3f, %.3f, %.3f) assigned to "
                "cell %zu "
                "with bounds [%.3f-%.3f, %.3f-%.3f, %.3f-%.3f]",
                gp->loc[0], gp->loc[1], gp->loc[2], cid, cell->loc[0],
                cell->loc[0] + cell->width[0], cell->loc[1],
                cell->loc[1] + cell->width[1], cell->loc[2],
                cell->loc[2] + cell->width[2]);
        errors++;
      }
    }
  }

  if (errors > 0) {
    error("[DEBUG] Found %d grid points incorrectly assigned to cells", errors);
  } else {
    message("[DEBUG] ✓ All grid points correctly assigned to cells");
  }
}

/**
 * @brief Validate that all useful cells have been correctly identified
 *
 * Checks that:
 * 1. All cells with grid points are marked as useful
 * 2. All neighbors of cells with grid points are marked as useful
 */
void validateUsefulCells(Simulation *sim, [[maybe_unused]] Grid *grid) {
  message("[DEBUG] Validating useful cell flagging...");

  int errors = 0;
  std::vector<Cell> &cells = sim->cells;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    // Check 1: Cells with grid points must be useful
    if (cell->grid_points.size() > 0 && !cell->is_useful) {
      message("[DEBUG] ERROR: Cell %zu has %zu grid points but is_useful = "
              "false",
              cid, cell->grid_points.size());
      errors++;
    }

    // Check 2: Neighbors of cells with grid points must be useful
    if (cell->grid_points.size() > 0) {
      for (Cell *neighbor : cell->neighbours) {
        if (!neighbor->is_useful) {
          message(
              "[DEBUG] ERROR: Cell %zu has grid points but its neighbor "
              "(cell at %.3f,%.3f,%.3f) is not marked as useful",
              cid, neighbor->loc[0], neighbor->loc[1], neighbor->loc[2]);
          errors++;
        }
      }
    }
  }

  // Count total useful cells and cells with grid points
  int useful_count = 0;
  int cells_with_gps = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].is_useful)
      useful_count++;
    if (cells[cid].grid_points.size() > 0)
      cells_with_gps++;
  }

  message("[DEBUG] Useful cells: %d, Cells with grid points: %d", useful_count,
          cells_with_gps);

  if (errors > 0) {
    error("[DEBUG] Found %d useful cell flagging errors", errors);
  } else {
    message("[DEBUG] ✓ All useful cells correctly flagged");
  }
}

/**
 * @brief Validate that particles are in the correct cells
 *
 * Checks that each particle's position is within the bounds of its assigned
 * cell.
 */
void validateParticleCellAssignment(Simulation *sim) {
  message("[DEBUG] Validating particle to cell assignments...");

  int errors = 0;
  std::vector<Cell> &cells = sim->cells;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    for (Particle *part : cell->particles) {
      // Check if particle is within cell bounds
      bool inside_x = (part->pos[0] >= cell->loc[0]) &&
                      (part->pos[0] < cell->loc[0] + cell->width[0]);
      bool inside_y = (part->pos[1] >= cell->loc[1]) &&
                      (part->pos[1] < cell->loc[1] + cell->width[1]);
      bool inside_z = (part->pos[2] >= cell->loc[2]) &&
                      (part->pos[2] < cell->loc[2] + cell->width[2]);

      if (!inside_x || !inside_y || !inside_z) {
        message("[DEBUG] ERROR: Particle at (%.3f, %.3f, %.3f) assigned to "
                "cell %zu "
                "with bounds [%.3f-%.3f, %.3f-%.3f, %.3f-%.3f]",
                part->pos[0], part->pos[1], part->pos[2], cid, cell->loc[0],
                cell->loc[0] + cell->width[0], cell->loc[1],
                cell->loc[1] + cell->width[1], cell->loc[2],
                cell->loc[2] + cell->width[2]);
        errors++;
        if (errors > 10) { // Limit error spam
          message("[DEBUG] ... (suppressing further particle assignment "
                  "errors)");
          break;
        }
      }
    }
    if (errors > 10)
      break;
  }

  if (errors > 0) {
    error("[DEBUG] Found %d particles incorrectly assigned to cells", errors);
  } else {
    message("[DEBUG] ✓ All particles correctly assigned to cells");
  }
}

/**
 * @brief Count particles within radius of a grid point (brute force check)
 */
int bruteForceCountParticles(GridPoint *grid_point, Simulation *sim,
                             double radius) {
  int count = 0;
  double radius2 = radius * radius;
  double *dim = sim->dim;

  std::vector<Cell> &cells = sim->cells;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    Cell *cell = &cells[cid];

    for (Particle *part : cell->particles) {
      double dx = nearest(part->pos[0] - grid_point->loc[0], dim[0]);
      double dy = nearest(part->pos[1] - grid_point->loc[1], dim[1]);
      double dz = nearest(part->pos[2] - grid_point->loc[2], dim[2]);
      double r2 = dx * dx + dy * dy + dz * dz;

      if (r2 <= radius2) {
        count++;
      }
    }
  }

  return count;
}

/**
 * @brief Check if grid points can find any particles within kernel radii
 *
 * For ALL grid points, performs a brute force search to count particles within
 * each kernel radius, stores the counts, then compares to what the gridder found.
 * In debug mode, these brute force counts will be written to the output file.
 */
void validateGridPointsHaveParticles(Simulation *sim, Grid *grid) {
  message("[DEBUG] Computing brute force particle counts for all grid points and all kernels...");
  message("[DEBUG] This will be written to output file for comparison.");

  int total_mismatches = 0;
  int total_empty_kernels = 0;
  int total_checks = 0;

  // For ALL grid points (not sampled), compute brute force counts for ALL kernels
  for (size_t i = 0; i < grid->grid_points.size(); i++) {
    GridPoint *gp = &grid->grid_points[i];

    // Compute brute force for each kernel radius
    for (double kernel_rad : grid->kernel_radii) {
      int brute_count = bruteForceCountParticles(gp, sim, kernel_rad);
      int gridder_count = gp->getCount(kernel_rad);

      // Store the brute force count
      gp->setBruteForceCount(kernel_rad, brute_count);

      total_checks++;

      if (brute_count == 0) {
        total_empty_kernels++;
      }

      if (brute_count != gridder_count) {
        total_mismatches++;
      }
    }
  }

  message("[DEBUG] Brute force validation complete:");
  message("[DEBUG]   - Checked %zu grid points × %d kernels = %d combinations",
          grid->grid_points.size(), grid->nkernels, total_checks);
  message("[DEBUG]   - %d had no particles (expected for small kernels/sparse regions)",
          total_empty_kernels);
  message("[DEBUG]   - %d had count mismatches between brute force and gridder",
          total_mismatches);

  if (total_mismatches > 0) {
    message("[DEBUG] ╔═══════════════════════════════════════════════════════════════╗");
    message("[DEBUG] ║ WARNING: PARTICLE COUNT MISMATCHES DETECTED                   ║");
    message("[DEBUG] ╠═══════════════════════════════════════════════════════════════╣");
    message("[DEBUG] ║ Found %5d mismatches between octree and brute force        ║", total_mismatches);
    message("[DEBUG] ║                                                               ║");
    message("[DEBUG] ║ THE RESULTS IN THE OUTPUT FILE ARE INCORRECT!                ║");
    message("[DEBUG] ║                                                               ║");
    message("[DEBUG] ║ The output file contains both:                               ║");
    message("[DEBUG] ║   - GridPointCounts:                                         ║");
    message("[DEBUG] ║       Counts from octree traversal (INCORRECT - has bugs)    ║");
    message("[DEBUG] ║   - BruteForceGridPointCounts:                               ║");
    message("[DEBUG] ║       Counts from exhaustive search (CORRECT - ground truth) ║");
    message("[DEBUG] ║                                                               ║");
    message("[DEBUG] ║ Use BruteForceGridPointCounts for analysis.                  ║");
    message("[DEBUG] ║                                                               ║");
    message("[DEBUG] ║ This likely indicates a bug in periodic boundary handling    ║");
    message("[DEBUG] ║ or octree traversal, especially for grid points near box     ║");
    message("[DEBUG] ║ boundaries or with small kernel radii.                       ║");
    message("[DEBUG] ╚═══════════════════════════════════════════════════════════════╝");
  } else if (total_empty_kernels > total_checks / 2) {
    message("[DEBUG] WARNING: More than 50%% of grid point/kernel combinations "
            "have no particles nearby");
  } else {
    message("[DEBUG] ✓ Grid point particle counts match brute force validation");
  }
}

/**
 * @brief Validate octree structure integrity
 *
 * Recursively checks that:
 * 1. Child cells properly partition their parent
 * 2. Particle counts match between parent and children
 * 3. Grid point counts match between parent and children
 */
static void validateCellRecursive(Cell *cell, int *errors) {
  if (!cell->is_split)
    return;

  // Check particle count consistency
  size_t child_part_count = 0;
  for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
    if (cell->children[i] != nullptr) {
      child_part_count += cell->children[i]->part_count;
      validateCellRecursive(cell->children[i], errors);
    }
  }

  if (child_part_count != cell->part_count) {
    message(
        "[DEBUG] ERROR: Cell at (%.3f, %.3f, %.3f) has %zu particles but "
        "children have %zu total",
        cell->loc[0], cell->loc[1], cell->loc[2], cell->part_count,
        child_part_count);
    (*errors)++;
  }

  // Check grid point count consistency
  size_t child_gp_count = 0;
  for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
    if (cell->children[i] != nullptr) {
      child_gp_count += cell->children[i]->grid_points.size();
    }
  }

  if (child_gp_count != cell->grid_points.size()) {
    message("[DEBUG] ERROR: Cell at (%.3f, %.3f, %.3f) has %zu grid points "
            "but children have %zu total",
            cell->loc[0], cell->loc[1], cell->loc[2], cell->grid_points.size(),
            child_gp_count);
    (*errors)++;
  }
}

void validateOctreeStructure(Simulation *sim) {
  message("[DEBUG] Validating octree structure...");

  int errors = 0;
  std::vector<Cell> &cells = sim->cells;

  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    validateCellRecursive(&cells[cid], &errors);
  }

  if (errors > 0) {
    error("[DEBUG] Found %d octree structure errors", errors);
  } else {
    message("[DEBUG] ✓ Octree structure is valid");
  }
}

/**
 * @brief Check that file-based grid points are within simulation boundaries
 */
void validateFileGridPoints(Grid *grid, Simulation *sim) {
  if (!grid->grid_from_file)
    return;

  message("[DEBUG] Validating file-based grid points are within simulation "
          "boundaries...");

  int outside_count = 0;
  double *dim = sim->dim;

  for (GridPoint &gp : grid->grid_points) {
    if (gp.loc[0] < 0 || gp.loc[0] >= dim[0] || gp.loc[1] < 0 ||
        gp.loc[1] >= dim[1] || gp.loc[2] < 0 || gp.loc[2] >= dim[2]) {
      message("[DEBUG] ERROR: Grid point at (%.3f, %.3f, %.3f) is outside "
              "simulation box [0-%.3f, 0-%.3f, 0-%.3f]",
              gp.loc[0], gp.loc[1], gp.loc[2], dim[0], dim[1], dim[2]);
      outside_count++;
    }
  }

  if (outside_count > 0) {
    error("[DEBUG] Found %d grid points outside simulation boundaries",
          outside_count);
  } else {
    message("[DEBUG] ✓ All file-based grid points within simulation "
            "boundaries");
  }
}

/**
 * @brief Print detailed diagnostics about a specific grid point
 */
void diagnoseGridPoint(GridPoint *grid_point, Simulation *sim, Grid *grid) {
  message("[DEBUG] ==== Grid Point Diagnostic ====");
  message("[DEBUG] Location: (%.6f, %.6f, %.6f)", grid_point->loc[0],
          grid_point->loc[1], grid_point->loc[2]);

  // Find which cell it's in
  Cell *cell = getCellContainingPoint(grid_point->loc);
  message("[DEBUG] Assigned to cell at (%.3f, %.3f, %.3f) with width (%.3f, "
          "%.3f, %.3f)",
          cell->loc[0], cell->loc[1], cell->loc[2], cell->width[0],
          cell->width[1], cell->width[2]);
  message("[DEBUG] Cell has %zu particles, %zu grid points", cell->part_count,
          cell->grid_points.size());
  message("[DEBUG] Cell is_useful: %s, is_split: %s",
          cell->is_useful ? "true" : "false", cell->is_split ? "true" : "false");

  // Check neighbors
  message("[DEBUG] Cell has %zu neighbors", cell->neighbours.size());
  size_t total_neighbor_particles = 0;
  for (Cell *neighbor : cell->neighbours) {
    total_neighbor_particles += neighbor->part_count;
  }
  message("[DEBUG] Neighbors have %zu total particles", total_neighbor_particles);

  // Check particle counts for each kernel
  message("[DEBUG] Particle counts by kernel radius:");
  for (double radius : grid->kernel_radii) {
    // Get count from gridder
    int count = grid_point->getCount(radius);
    int brute_count = bruteForceCountParticles(grid_point, sim, radius);
    message("[DEBUG]   %.3f Mpc/h: gridder=%d, brute_force=%d", radius, count,
            brute_count);
  }

  message("[DEBUG] ================================");
}

#endif // DEBUGGING_CHECKS
