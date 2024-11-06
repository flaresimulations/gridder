
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
  std::shared_ptr<Cell> *cells = sim->cells;
  const double width[3] = {sim->width[0], sim->width[1], sim->width[2]};
  const int nr_cells = sim->nr_cells;
  const int cdim[3] = {sim->cdim[0], sim->cdim[1], sim->cdim[2]};
  const std::vector<int> counts = sim->cell_part_counts;

// Loop over the cells and create them, storing the counts for domain
// decomposition
#pragma omp parallel for
  for (int cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell location and width
    double loc[3] = {i * width[0], j * width[1], k * width[2]};

    // Create the cell
    std::shared_ptr<Cell> cell =
        std::make_shared<Cell>(loc, width, /*parent*/ nullptr);

    // Assign the particle count in this cell
    cell->part_count = counts[cid];

    // We need to set top outside the constructor
    cell->top = cell;

    // Add the cell to the cells vector
    cells[cid] = cell;
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

  // Loop over the cells attaching the pointers the neighbouring cells (taking
  // into account periodic boundary conditions)
#pragma omp parallel for
  for (int cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell
    std::shared_ptr<Cell> cell = cells[cid];

    // Loop over the neighbours
    int nid = 0;
    for (int ii = -nwalk; ii < nwalk + 1; ii++) {
      for (int jj = -nwalk; jj < nwalk + 1; jj++) {
        for (int kk = -nwalk; kk < nwalk + 1; kk++) {

          // Skip the cell itself
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          // Get the neighbour index (handling periodic boundary conditions)
          int iii = (i + ii + cdim[0]) % cdim[0];
          int jjj = (j + jj + cdim[1]) % cdim[1];
          int kkk = (k + kk + cdim[2]) % cdim[2];
          int cjd = iii * cdim[1] * cdim[2] + jjj * cdim[2] + kkk;

          // Attach the neighbour to the cell
          cell->neighbours.push_back(cells[cjd]);
        }
      }
    }
  }
}
