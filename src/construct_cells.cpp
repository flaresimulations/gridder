
// Standard includes
#include <cmath>
#include <memory>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "simulation.hpp"

void getTopCells(Simulation *sim, Grid *grid) {
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
  // the maximum distance at which we will need to consider another cell
  const int nwalk = std::ceil(grid->max_kernel_radius / width[0]) + 1;
  int nwalk_upper = nwalk;
  int nwalk_lower = nwalk;

  // If nwalk is greater than the number of cells in the simulation, we need
  // to walk out to the edge of the simulation
  if (nwalk > cdim[0] / 2) {
    nwalk_upper = cdim[0] / 2;
    nwalk_lower = cdim[0] / 2;
  }

  message("Looking for neighbours within %d cells", nwalk);

  // Calculate maximum neighbors and reserve space
  const int max_neighbors =
      (2 * nwalk + 1) * (2 * nwalk + 1) * (2 * nwalk + 1) -
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

    // Loop over the neighbours
    for (int ii = -nwalk_lower; ii < nwalk_upper + 1; ii++) {
      for (int jj = -nwalk_lower; jj < nwalk_upper + 1; jj++) {
        for (int kk = -nwalk_lower; kk < nwalk_upper + 1; kk++) {

          // Skip the cell itself
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          // Get the neighbour index (handling periodic boundary conditions)
          int iii = (i + ii + cdim[0]) % cdim[0];
          int jjj = (j + jj + cdim[1]) % cdim[1];
          int kkk = (k + kk + cdim[2]) % cdim[2];
          int cjd = iii * cdim[1] * cdim[2] + jjj * cdim[2] + kkk;

          // Attach the neighbour to the cell
          cell->neighbours.push_back(&cells[cjd]);
        }
      }
    }
  }
}

/**
 * @brief Split the top level cells to create the cell tree.
 *
 * @param cells The top level cells
 */
void splitCells(Simulation *sim) {
  // Unpack the cells
  const size_t nr_cells = sim->nr_cells;
  std::vector<Cell> &cells = sim->cells;

  // Loop over the cells and split them
#pragma omp parallel for
  for (size_t cid = 0; cid < nr_cells; cid++) {

#ifdef WITH_MPI
    // Get the metadata instance for MPI rank checking
    Metadata *metadata = &Metadata::getInstance();
    // Skip cells that aren't on this rank and aren't proxies
    if (cells[cid].rank != metadata->rank || !cells[cid].is_proxy)
      continue;
#endif

    cells[cid].split();
  }
}
