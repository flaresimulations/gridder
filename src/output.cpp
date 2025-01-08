// Standard includes

// Local includes
#include "grid_point.hpp"
#include "hdf_io.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

void writeGridFileSerial(Simulation *sim, Grid *grid) {

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata->output_file;

  message("Writing grid data to %s", filename.c_str());

  // Unpack the cells
  std::shared_ptr<Cell> *cells = sim->cells;

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC);

  // Create the Header group and write out the metadata
  hdf5.createGroup("Header");
  hdf5.writeAttribute<int>("Header", "NGridPoint", grid->n_grid_points);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", sim->cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", sim->dim);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              grid->max_kernel_radius);
  hdf5.writeAttribute<double>("Header", "Redshift", sim->redshift);

  // Create the Grids group
  hdf5.createGroup("Grids");

  // Loop over cells and collect how many grid points we have in each cell
  std::vector<int> grid_point_counts(sim->nr_cells, 0);
  for (int cid = 0; cid < sim->nr_cells; cid++) {
    grid_point_counts[cid] = cells[cid]->grid_points.size();
  }

  // Now we have the counts convert these to a start index for each cell so we
  // can use a cell look up table to find the grid points
  std::vector<int> grid_point_start(sim->nr_cells, 0);
  for (int cid = 1; cid < sim->nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Write out this cell lookup table
  std::array<hsize_t, 1> sim_cell_dims = {static_cast<hsize_t>(sim->nr_cells)};
  hdf5.createGroup("Cells");
  hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                            sim_cell_dims);
  hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                            sim_cell_dims);

  // Create a dataset we'll write the grid positions into
  std::array<hsize_t, 2> grid_point_positions_dims = {
      static_cast<hsize_t>(grid->n_grid_points), static_cast<hsize_t>(3)};
  hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                grid_point_positions_dims);

  // We only want to write the positions once so lets make a flag to ensure
  // we only do this once
  bool written_positions = false;

  // Loop over the kernels and write out the grids themselves
  for (double kernel_rad : grid->kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Create the kernel group
    hdf5.createGroup("Grids/" + kernel_name);

    // Create the grid point over densities dataset
    std::array<hsize_t, 1> grid_point_overdens_dims = {
        static_cast<hsize_t>(grid->n_grid_points)};
    hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                      "/GridPointOverDensities",
                                  grid_point_overdens_dims);

    // Write out the grid data cell by cell
    for (int cid = 0; cid < sim->nr_cells; cid++) {
      // Get the cell
      std::shared_ptr<Cell> cell = cells[cid];

      // Skip cells empty cells
      if (cell->grid_points.size() == 0)
        continue;

      // Get the start and end indices for this cell's grid points in the
      // global grid point array
      int start = grid_point_start[cid];

      // Create the output array for this cell
      std::vector<double> cell_grid_ovdens(grid_point_counts[cid], 0.0);
      std::vector<double> cell_grid_pos(grid_point_counts[cid] * 3, 0.0);

      // Loop over grid points and populate this cell's slices
      for (const std::shared_ptr<GridPoint> &gp : cell->grid_points) {
        // Get the over density for this grid point
        cell_grid_ovdens.push_back(gp->getOverDensity(kernel_rad, sim));

        // If we need to get the positions then do so now
        if (!written_positions) {
          cell_grid_pos.push_back(gp->loc[0]);
          cell_grid_pos.push_back(gp->loc[1]);
          cell_grid_pos.push_back(gp->loc[2]);
        }
      }

      // Write out the grid point over densities for this cell
      hdf5.writeDatasetSlice<double, 1>(
          "Grids/" + kernel_name + "/GridPointOverDensities", cell_grid_ovdens,
          {static_cast<hsize_t>(start)},
          {static_cast<hsize_t>(grid_point_counts[cid])});

      // If we haven't written the grid point positions yet then do so now
      if (!written_positions) {
        hdf5.writeDatasetSlice<double, 2>(
            "Grids/GridPointPositions", cell_grid_pos,
            {static_cast<hsize_t>(start), 0},
            {static_cast<hsize_t>(grid_point_counts[cid]), 3});
      }

    } // End of cell Loop

    // Once we've got here we know we've written the grid point positions
    written_positions = true;

  } // End of kernel loop

  // Close the HDF5 file
  hdf5.close();
}

#ifdef WITH_MPI
void writeGridFileParallel(Simulation *sim, Grid *grid) {

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata->output_file;

  message("Writing grid data to %s", filename.c_str());

  // Unpack the cells
  std::shared_ptr<Cell> *cells = sim->cells;

  // How many cells do we have before the first on our rank?
  const int nr_cells_before = 0;
  for (int cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid]->rank == metadata->rank)
      break;
    nr_cells_before++;
  }

  // Define the cell slice for this rank
  const int first_local_cell = nr_cells_before;
  const int nr_local_cells = metadata->nr_local_cells;
  const int last_local_cell = first_local_cell + nr_local_cells;

  // Loop over cells and collect how many grid points we have in each cell
  std::vector<int> local_grid_point_counts(sim->nr_cells, 0);
  for (int cid = first_local_cell; cid < last_local_cell; cid++) {
    local_grid_point_counts[cid] = cells[cid]->grid_points.size();
  }

  // Collect the grid point counts from all ranks
  std::vector<int> grid_point_counts(sim->nr_cells, 0);
  MPI_Allreduce(local_grid_point_counts.data(), grid_point_counts.data(),
                sim->nr_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Now we have the counts convert these to a start index for each cell so
  // we can use a cell look up table to find the grid points
  std::vector<int> grid_point_start(sim->nr_cells, 0);
  for (int cid = 1; cid < sim->nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Loop over ranks, we are going to do the I/O in the stupidist one-by-one
  // simple way
  for (int rank = 0; rank < metadata->size; rank++) {

    // Wait for everyone to get here on each loop
    MPI_Barrier(MPI_COMM_WORLD);

    // If this isn't our rank then continue
    if (rank != metadata->rank)
      continue;

    // Create a new HDF5 file
    HDF5Helper hdf5(filename, H5F_ACC_TRUNC);

    // Create the Header group and write out the metadata (only rank 0 does
    // this)
    if (rank == 0) {
      hdf5.createGroup("Header");
      hdf5.writeAttribute<int>("Header", "NGridPoint", grid->n_grid_points);
      hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", sim->cdim);
      hdf5.writeAttribute<double[3]>("Header", "BoxSize", sim->dim);
      hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                                  grid->max_kernel_radius);
      hdf5.writeAttribute<double>("Header", "Redshift", sim->redshift);
    }

    // Create the Grids group
    if (rank == 0) {
      hdf5.createGroup("Grids");
    }

    // Write out this cell lookup table
    std::array<hsize_t, 1> sim_cell_dims = {
        static_cast<hsize_t>(sim->nr_cells)};
    if (rank == 0) {
      hdf5.createGroup("Cells");
      hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                                sim_cell_dims);
      hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                                sim_cell_dims);
    }

    // Create a dataset we'll write the grid positions into
    if (rank == 0) {
      std::array<hsize_t, 2> grid_point_positions_dims = {
          static_cast<hsize_t>(grid->n_grid_points), static_cast<hsize_t>(3)};
      hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                    grid_point_positions_dims);
    }

    // We only want to write the positions once so lets make a flag to ensure
    // we only do this once
    bool written_positions = false;

    // Loop over the kernels and write out the grids themselves
    for (double kernel_rad : grid->kernel_radii) {
      std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

      if (rank == 0) {
        // Create the kernel group
        hdf5.createGroup("Grids/" + kernel_name);

        // Create the grid point over densities dataset
        std::array<hsize_t, 1> grid_point_overdens_dims = {
            static_cast<hsize_t>(grid->n_grid_points)};
        hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                          "/GridPointOverDensities",
                                      grid_point_overdens_dims);
      }

      // Write out the grid data cell by cell
      for (int cid = 0; cid < sim->nr_cells; cid++) {
        // Get the cell
        std::shared_ptr<Cell> cell = cells[cid];

        // Skip cells empty cells
        if (cell->grid_points.size() == 0)
          continue;

        // Get the start and end indices for this cell's grid points in the
        // global grid point array
        int start = grid_point_start[cid];

        // Create the output array for this cell
        std::vector<double> cell_grid_ovdens(grid_point_counts[cid], 0.0);
        std::vector<double> cell_grid_pos(grid_point_counts[cid] * 3, 0.0);

        // Loop over grid points and populate this cell's slices
        for (const std::shared_ptr<GridPoint> &gp : cell->grid_points) {
          // Get the over density for this grid point
          cell_grid_ovdens.push_back(gp->getOverDensity(kernel_rad, sim));

          // If we need to get the positions then do so now
          if (!written_positions) {
            cell_grid_pos.push_back(gp->loc[0]);
            cell_grid_pos.push_back(gp->loc[1]);
            cell_grid_pos.push_back(gp->loc[2]);
          }
        }

        // Write out the grid point over densities for this cell
        hdf5.writeDatasetSlice<double, 1>(
            "Grids/" + kernel_name + "/GridPointOverDensities",
            cell_grid_ovdens, {static_cast<hsize_t>(start)},
            {static_cast<hsize_t>(grid_point_counts[cid])});

        // If we haven't written the grid point positions yet then do so now
        if (!written_positions) {
          hdf5.writeDatasetSlice<double, 2>(
              "Grids/GridPointPositions", cell_grid_pos,
              {static_cast<hsize_t>(start), 0},
              {static_cast<hsize_t>(grid_point_counts[cid]), 3});
        }

      } // End of cell Loop

      // Once we've got here we know we've written the grid point positions
      written_positions = true;

    } // End of kernel loop

    // Close the HDF5 file
    hdf5.close();
  }
}
#endif
