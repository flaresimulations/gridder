
// Standard includes
#include <cmath>
#include <memory>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "simulation.hpp"

void getTopCells(Simulation *sim, Grid *grid) {

  tic();

  // Unpack the simulation information we need
  std::vector<Cell> &cells = sim->cells;
  const double width[3] = {sim->width[0], sim->width[1], sim->width[2]};
  const size_t nr_cells = sim->nr_cells;
  const int cdim[3] = {sim->cdim[0], sim->cdim[1], sim->cdim[2]};
  const std::vector<int> counts = sim->cell_part_counts;

// Loop over the cells and create them, storing the counts for domain
// decomposition
#pragma omp parallel for
  for (size_t cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell location and width
    double loc[3] = {i * width[0], j * width[1], k * width[2]};

    // Initialize the cell in-place
    cells[cid] = Cell(loc, width, /*parent*/ nullptr);

    // Assign the particle count in this cell
    cells[cid].part_count = counts[cid];

    // We need to set top outside the constructor
    cells[cid].top = &cells[cid];
  }

  // Now the top level cells are made we can attached the pointers to
  // neighbouring cells (this simplifies boilerplate elsewhere)

  // How many cells do we need to walk out for the biggest kernel? This is
  // the maximum distance at which we will need to consider another cell.
  // We compute this separately for each dimension since cell widths and
  // grid dimensions may differ.
  int nwalk[3];
  for (int dim = 0; dim < 3; dim++) {
    nwalk[dim] = std::ceil(grid->max_kernel_radius / width[dim]) + 1;

    // Clamp to half the grid dimension to prevent duplicate neighbors
    // through periodic wrapping. If we walk more than cdim/2, we'll
    // encounter the same cell from multiple periodic images.
    if (nwalk[dim] > cdim[dim] / 2) {
      nwalk[dim] = cdim[dim] / 2;
    }
  }

  message("Looking for neighbours within [%d, %d, %d] cells",
          nwalk[0], nwalk[1], nwalk[2]);

  // Calculate maximum neighbors (use the maximum nwalk for reservation)
  const int max_nwalk = std::max({nwalk[0], nwalk[1], nwalk[2]});
  const int max_neighbors =
      (2 * max_nwalk + 1) * (2 * max_nwalk + 1) * (2 * max_nwalk + 1) -
      1; // -1 excludes self

  // Loop over the cells attaching the pointers the neighbouring cells (taking
  // into account periodic boundary conditions)
#pragma omp parallel for
  for (size_t cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell
    Cell *cell = &cells[cid];

    // Reserve space for neighbors
    cell->neighbours.reserve(max_neighbors);

    // Loop over the neighbours using dimension-specific nwalk values
    // The nwalk values are already clamped to cdim/2, preventing duplicates
    for (int ii = -nwalk[0]; ii <= nwalk[0]; ii++) {
      for (int jj = -nwalk[1]; jj <= nwalk[1]; jj++) {
        for (int kk = -nwalk[2]; kk <= nwalk[2]; kk++) {

          // Skip the cell itself
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          // Get the neighbour index (handling periodic boundary conditions)
          int iii = (i + ii + cdim[0]) % cdim[0];
          int jjj = (j + jj + cdim[1]) % cdim[1];
          int kkk = (k + kk + cdim[2]) % cdim[2];
          int cjd = iii * cdim[1] * cdim[2] + jjj * cdim[2] + kkk;

          // Skip if this wraps back to the cell itself
          // (can happen with periodic boundaries in small boxes)
          if (cjd == static_cast<int>(cid))
            continue;

          // Attach the neighbour to the cell
          cell->neighbours.push_back(&cells[cjd]);
        }
      }
    }
  }

  toc("Creating top level cells");
}

/**
 * @brief Split the top level cells to create the cell tree.
 *
 * @param cells The top level cells
 */
void splitCells(Simulation *sim) {

  tic();

#ifdef WITH_MPI
  // Get the metadata instance for MPI rank checking
  Metadata *metadata = &Metadata::getInstance();

  // In MPI mode, use the locally_useful_cells lookup vector
  // Only split cells on this rank
  std::vector<Cell *> cells_to_split;
  cells_to_split.reserve(sim->locally_useful_cells.size());

  for (Cell *cell : sim->locally_useful_cells) {
    if (cell->rank == metadata->rank) {
      cells_to_split.push_back(cell);
    }
  }

  // Loop over locally useful cells on this rank and split them
#pragma omp parallel for
  for (size_t i = 0; i < cells_to_split.size(); i++) {
    cells_to_split[i]->split();
  }

  message("Rank %d: Split %zu locally useful cells", metadata->rank,
          cells_to_split.size());
#else
  // In serial mode, use the useful_cells lookup vector
  // Loop over useful cells and split them
#pragma omp parallel for
  for (size_t i = 0; i < sim->useful_cells.size(); i++) {
    sim->useful_cells[i]->split();
  }

  message("Split %zu useful cells", sim->useful_cells.size());
#endif

  message("Maximum depth in the tree: %d", sim->max_depth);

  toc("Splitting cells");
}
