// Standard includes
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

namespace {
struct KernelTraversalContext {
  std::vector<double> sorted_radii2;
  std::vector<size_t> original_indices;
};
} // namespace

/**
 * @brief Function to assign particles to a grid point.
 *
 * @param cell The cell to assign particles to grid points within.
 * @param grid_point The grid point to assign particles to.
 * @param kernels Sorted kernel radii and their original accumulator indices.
 * @param kernel_begin First unresolved radius in the sorted arrays.
 * @param kernel_end One past the last unresolved radius.
 */
static void addPartsToGridPoint(Cell *cell, GridPoint *grid_point,
                                const KernelTraversalContext &kernels,
                                const size_t kernel_begin,
                                const size_t kernel_end) {

  // Get the boxsize from the metadata
  Metadata *metadata = &Metadata::getInstance();
  Simulation *sim = metadata->sim;
  double *dim = sim->dim;

  // Range-mode leaves access physically contiguous properties directly. The
  // indexed fallback retains the existing indirect lookup.
  const size_t stored_particle_count =
      sim->particle_ranges_enabled ? cell->part_count : cell->particles.size();
  for (size_t p = 0; p < stored_particle_count; p++) {
    const ParticleIndex part = sim->particle_ranges_enabled
                                   ? cell->particle_offset + p
                                   : cell->particles[p];
    const double *part_pos = sim->particlePosition(part);

    // Get the distance between the particle and the grid point
    double dx = nearest(part_pos[0] - grid_point->loc[0], dim[0]);
    double dy = nearest(part_pos[1] - grid_point->loc[1], dim[1]);
    double dz = nearest(part_pos[2] - grid_point->loc[2], dim[2]);
    double r2 = dx * dx + dy * dy + dz * dz;

    // The containing radii form a suffix because radii are sorted. Locate the
    // first match once, then update it and every larger unresolved kernel.
    const auto first_containing = std::lower_bound(
        kernels.sorted_radii2.begin() + kernel_begin,
        kernels.sorted_radii2.begin() + kernel_end, r2);
    const size_t first_index =
        static_cast<size_t>(first_containing - kernels.sorted_radii2.begin());
    if (first_index == kernel_end)
      continue;
    const double particle_mass = sim->particleMass(part);
    for (size_t k = first_index; k < kernel_end; k++)
      grid_point->add_particle(particle_mass, kernels.original_indices[k]);
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
 * @param kernels Sorted kernel radii and original accumulator indices
 * @param kernel_begin First unresolved radius
 * @param kernel_end One past the last unresolved radius
 */
static void recursivePairPartsToPoints(Cell *cell, Cell *other,
                                       const KernelTraversalContext &kernels,
                                       const size_t kernel_begin,
                                       const size_t kernel_end) {

  // Ensure we have grid points, otherwise there's nothing to add to
  if (cell->grid_points.size() == 0 || kernel_begin == kernel_end)
    return;

  // Ensure the other cell has particles, otherwise there's nothing to add
  if (other->part_count == 0)
    return;

  // If we have more than one grid point recurse (we can always do this since
  // the cell tree was constructed such that the leaves have only 1 grid point)
  if (cell->grid_points.size() > 1) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursivePairPartsToPoints(cell->children[i], other, kernels,
                                 kernel_begin, kernel_end);
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
  GridPoint *grid_point = cell->grid_points[0];

  // Outside decisions form a prefix of the sorted radii. Discard that prefix.
  size_t overlap_begin = kernel_begin;
  while (overlap_begin < kernel_end &&
         other->outsideKernel(grid_point,
                              kernels.sorted_radii2[overlap_begin]))
    overlap_begin++;
  if (overlap_begin == kernel_end)
    return;

  // Whole-cell acceptance forms a suffix. Accumulate that suffix immediately;
  // only the middle range of partially overlapping radii needs more work.
  size_t inside_begin = overlap_begin;
  while (inside_begin < kernel_end &&
         !other->inKernel(grid_point, kernels.sorted_radii2[inside_begin]))
    inside_begin++;
  for (size_t k = inside_begin; k < kernel_end; k++) {
    grid_point->add_cell(other->part_count, other->mass,
                         kernels.original_indices[k]);
  }
  if (overlap_begin == inside_begin)
    return;

  // Internal cells retain aggregate count and mass but release their particle
  // indices after splitting. For a partial overlap, always descend until an
  // unsplit leaf supplies the indices that need explicit distance checks.
  if (other->is_split) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursivePairPartsToPoints(cell, other->children[i], kernels,
                                 overlap_begin, inside_begin);
    }
    return;
  }

  // Ok, we can't just add the whole cell to the grid point, instead check
  // the particles in the other cell
  addPartsToGridPoint(other, grid_point, kernels, overlap_begin, inside_begin);
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
 * @param kernels Sorted kernel radii and original accumulator indices.
 * @param kernel_begin First unresolved radius.
 * @param kernel_end One past the last unresolved radius.
 */
static void recursiveSelfPartsToPoints(Cell *cell,
                                       const KernelTraversalContext &kernels,
                                       const size_t kernel_begin,
                                       const size_t kernel_end) {

  // Ensure we have grid points and particles
  if (cell->grid_points.size() == 0 || cell->part_count == 0 ||
      kernel_begin == kernel_end)
    return;

  // Split cells do not retain particle indices. Recurse even when this cell
  // contains only one grid point so self interactions are gathered from its
  // particle-bearing children and their siblings.
  if (cell->is_split) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      recursiveSelfPartsToPoints(cell->children[i], kernels, kernel_begin,
                                 kernel_end);

      // And do the pair assignment
      for (int j = 0; j < Cell::OCTREE_CHILDREN; j++) {
        if (i == j)
          continue;
        recursivePairPartsToPoints(cell->children[i], cell->children[j],
                                   kernels, kernel_begin, kernel_end);
      }
    }
  } else {
    if (cell->grid_points.size() > 1) {
      error("We shouldn't be able to find a leaf with more than 1 grid point "
            "(leaf->grid_points.size()=%d",
            cell->grid_points.size());
    }

    GridPoint *grid_point = cell->grid_points[0];
    const double cell_diag = cell->width[0] * cell->width[0] +
                             cell->width[1] * cell->width[1] +
                             cell->width[2] * cell->width[2];

    // Preserve the original self-cell shortcut: radii at least as large as
    // the cell diagonal contain every particle because the grid point lies in
    // this leaf. Only smaller radii require explicit particle distances.
    const auto first_containing_cell = std::lower_bound(
        kernels.sorted_radii2.begin() + kernel_begin,
        kernels.sorted_radii2.begin() + kernel_end, cell_diag);
    const size_t inside_begin = static_cast<size_t>(
        first_containing_cell - kernels.sorted_radii2.begin());
    for (size_t k = inside_begin; k < kernel_end; k++) {
      grid_point->add_cell(cell->part_count, cell->mass,
                           kernels.original_indices[k]);
    }
    if (kernel_begin < inside_begin) {
      addPartsToGridPoint(cell, grid_point, kernels, kernel_begin,
                          inside_begin);
    }
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

  tic();

  KernelTraversalContext kernels;
  kernels.original_indices.resize(grid->kernel_radii.size());
  std::iota(kernels.original_indices.begin(), kernels.original_indices.end(),
            size_t{0});
  std::stable_sort(
      kernels.original_indices.begin(), kernels.original_indices.end(),
      [&](const size_t lhs, const size_t rhs) {
        const double lhs_radius = grid->kernel_radii[lhs];
        const double rhs_radius = grid->kernel_radii[rhs];
        return lhs_radius * lhs_radius < rhs_radius * rhs_radius;
      });
  kernels.sorted_radii2.reserve(grid->kernel_radii.size());
  for (size_t kernel_index : kernels.original_indices) {
    const double radius = grid->kernel_radii[kernel_index];
    kernels.sorted_radii2.push_back(radius * radius);
  }

  if (kernels.sorted_radii2.empty()) {
    toc("Computing kernel masses");
    return;
  }

  message("Using one fused octree traversal for %zu kernel radii",
          kernels.sorted_radii2.size());

#ifdef WITH_MPI
  // Get the metadata instance for MPI rank checking
  Metadata *metadata = &Metadata::getInstance();

  // Build a list of local useful cells for efficient iteration
  // (only cells on this rank)
  std::vector<Cell *> local_useful_cells;
  local_useful_cells.reserve(sim->locally_useful_cells.size());
  for (Cell *cell : sim->locally_useful_cells) {
    if (cell->rank == metadata->rank) {
      local_useful_cells.push_back(cell);
    }
  }

  // Loop over the local cells only
#pragma omp parallel for
  for (size_t i = 0; i < local_useful_cells.size(); i++) {
    Cell *cell = local_useful_cells[i];
#else
  // In serial mode, use the useful_cells lookup vector directly
#pragma omp parallel for
  for (size_t i = 0; i < sim->useful_cells.size(); i++) {
    Cell *cell = sim->useful_cells[i];
#endif

    // Traverse the tree once for all radii, preserving the original kernel
    // accumulator indices through the sorted traversal context.
    recursiveSelfPartsToPoints(cell, kernels, 0, kernels.sorted_radii2.size());

    for (Cell *neighbour : cell->neighbours) {
      recursivePairPartsToPoints(cell, neighbour, kernels, 0,
                                 kernels.sorted_radii2.size());
    }
  }
  toc("Computing kernel masses");
}
