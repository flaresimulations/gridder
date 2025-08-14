// Standard includes
#include <cmath>
#include <memory>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

/**
 * @brief Function to assign particles to a grid point.
 *
 * @param cell The cell to assign particles to grid points within.
 * @param grid_point The grid point to assign particles to.
 * @param kernel_rad The kernel radius.
 * @param kernel_rad2 The squared kernel radius.
 */
static void addPartsToGridPoint(Cell* cell, GridPoint* grid_point,
                                const double kernel_rad,
                                const double kernel_rad2) {

  // Get the boxsize from the metadata
  Metadata *metadata = &Metadata::getInstance();
  double *dim = metadata->sim->dim;

  // Loop over the particles in the cell and assign them to the grid point
  for (size_t p = 0; p < cell->part_count; p++) {
    Particle* part = cell->particles[p];

    // Get the distance between the particle and the grid point
    double dx = nearest(part->pos[0] - grid_point->loc[0], dim[0]);
    double dy = nearest(part->pos[1] - grid_point->loc[1], dim[1]);
    double dz = nearest(part->pos[2] - grid_point->loc[2], dim[2]);
    double r2 = dx * dx + dy * dy + dz * dz;

    // If the particle is within the kernel radius of the grid point then
    // assign it
    if (r2 <= kernel_rad2) {
      grid_point->add_particle(part, kernel_rad);
    }
  }
}

/**
 * @brief Function to assign particles to grid points within a cell.
 *
 * This function handles pairs of cells.
 *
 * We only check a grid point when we reach the leaves of the cell tree. This
 * is where a cell only contains a single grid point.
 *
 * If a whole cell is within the kernel radius of a grid point then the entire
 * cell is added to the grid point. If only part of the cell overlaps with the
 * kernel then we loop over the particles checking. If the cell is not within
 * the kernel radius then we exit.
 *
 * @param cell The cell to assign particles to grid points within
 * @param other The other cell to assign particles from
 * @param kernel_rad The kernel radius
 * @param kernel_rad2 The squared kernel radius
 */
static void recursivePairPartsToPoints(Cell* cell,
                                       Cell* other,
                                       const double kernel_rad,
                                       const double kernel_rad2) {

  // Ensure we have grid points, otherwise there's nothing to add to
  if (cell->grid_points.size() == 0)
    return;

  // Ensure the other cell has particles, otherwise there's nothing to add
  if (other->part_count == 0)
    return;

  // If we have more than one grid point recurse (we can always do this since
  // the cell tree was constructed such that the leaves have only 1 grid point)
  if (cell->grid_points.size() > 1) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursivePairPartsToPoints(cell->children[i], other, kernel_rad,
                                 kernel_rad2);
    }
    return;
  }

  // Ensure we only have 1 grid point now we are in a leaf
  if (cell->grid_points.size() > 1) {
    error("We shouldn't be able to find a leaf with more than 1 grid point "
          "(leaf->grid_points.size()=%d",
          cell->grid_points.size());
  }

  // Get the single grid point in this leaf
  GridPoint* grid_point = cell->grid_points[0];

  // Early exit if the cells are too far apart.
  if (other->outsideKernel(grid_point, kernel_rad2))
    return;

  // Can we just add the whole cell to the grid point?
  if (other->inKernel(grid_point, kernel_rad2)) {
    grid_point->add_cell(other->part_count, other->mass, kernel_rad);
    return;
  }

  // Get an instance of the metadata
  Metadata &metadata = Metadata::getInstance();

  // If the other cell is split then we need to recurse over the children before
  // trying to add the particles
  if (other->is_split && other->part_count > metadata.max_leaf_count) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursivePairPartsToPoints(cell, other->children[i], kernel_rad,
                                 kernel_rad2);
    }
    return;
  }

  // Ok, we can't just add the whole cell to the grid point, instead check
  // the particles in the other cell
  addPartsToGridPoint(other, grid_point, kernel_rad, kernel_rad2);
}

/**
 * @brief Function to assign particles to grid points within a cell.
 *
 * This function handles particles within the same cell as the grid point.
 *
 * We only check a grid point when we reach the leaves of the cell tree. This
 * is where a cell only contains a single grid point.
 *
 * @param cell The cell to assign particles to grid points within.
 * @param kernel_rad The kernel radius.
 * @param kernel_rad2 The squared kernel radius.
 */
static void recursiveSelfPartsToPoints(Cell* cell,
                                       const double kernel_rad,
                                       const double kernel_rad2) {

  // Ensure we have grid points and particles
  if (cell->grid_points.size() == 0 || cell->part_count == 0)
    return;

  // If the cell is split then we need to recurse over the children
  if (cell->is_split && cell->grid_points.size() > 1) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursiveSelfPartsToPoints(cell->children[i], kernel_rad, kernel_rad2);

      // And do the pair assignment
      for (int j = 0; j < Cell::OCTREE_CHILDREN; j++) {
        if (i == j)
          continue;
        recursivePairPartsToPoints(cell->children[i], cell->children[j],
                                   kernel_rad, kernel_rad2);
      }
    }
  } else {

    // Ensure we only have 1 grid point now we are in a leaf
    if (cell->grid_points.size() > 1) {
      error("We shouldn't be able to find a leaf with more than 1 grid point "
            "(leaf->grid_points.size()=%d",
            cell->grid_points.size());
    }

    // Get the single grid point in this leaf
    GridPoint &grid_point = *cell->grid_points[0];

    // If the diagonal of the cell is less than the kernel radius then we can
    // just add the whole cell to the grid point since the entire cell is
    // within the kernel radius
    const double cell_diag = cell->width[0] * cell->width[0] +
                             cell->width[1] * cell->width[1] +
                             cell->width[2] * cell->width[2];
    if (cell_diag <= kernel_rad2) {
      grid_point.add_cell(cell->part_count, cell->mass, kernel_rad);
      return;
    }

    // Associate particles to the single grid point
    addPartsToGridPoint(cell, cell->grid_points[0], kernel_rad, kernel_rad2);
  }
}

/**
 * @brief Function to assign particles to grid points.
 *
 * This is the top level function which will recurse through the cell tree
 * assigning particles to grid points within each kernel radius.
 *
 * @param sim Simulation object.
 * @param grid Grid object.
 */
void getKernelMasses(Simulation *sim, Grid *grid) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the cells
  std::vector<Cell>& cells = sim->cells;

  // Loop over the cells
#pragma omp parallel for
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {

    // Get the cell
    Cell* cell = &cells[cid];

    // Skip unuseful cells
    if (!cell->is_useful)
      continue;

#ifdef WITH_MPI
    // Skip cells that aren't on this rank
    if (cell->rank != metadata.rank)
      continue;
#endif

    // Loop over kernels
    for (double kernel_rad : grid->kernel_radii) {

      // Compute squared kernel radius
      double kernel_rad2 = kernel_rad * kernel_rad;

      // Recursively assign particles within a cell to the grid points within
      // the cell
      recursiveSelfPartsToPoints(cell, kernel_rad, kernel_rad2);

      // Recursively assign particles within any neighbours to the grid points
      // within a cell
      for (Cell* neighbour : cell->neighbours) {
        recursivePairPartsToPoints(cell, neighbour, kernel_rad, kernel_rad2);
      }
    }
  }
}
