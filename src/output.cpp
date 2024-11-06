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
/**
 * @brief Writes grid data to an HDF5 file in parallel using MPI
 *
 * This function writes simulation grid data in parallel across multiple MPI
 * ranks, with each rank writing only the data it owns. The function leverages
 * HDF5 parallel I/O capabilities, using collective operations to ensure data
 * consistency and performance.
 *
 * Each rank:
 * - Aggregates its local grid point counts, which are then reduced across all
 *   ranks so each has a consistent view of the total grid.
 * - Writes data to dedicated portions of the HDF5 file, minimizing contention.
 * - Writes global metadata (attributes) only from rank 0 to avoid duplication.
 *
 * @param cells Pointer to an array of Cell objects representing the simulation
 * grid
 * @param comm MPI communicator used for parallel I/O
 */
void writeGridFileParallel(std::shared_ptr<Cell> *cells, MPI_Comm comm) {

  // Retrieve global simulation metadata
  Metadata &metadata = Metadata::getInstance();

  // Define the output filename from metadata
  const std::string filename = metadata.output_file;
  message("Writing grid data to %s", filename.c_str());

  // Initialize the HDF5 file in parallel mode
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC, comm);

  // Only rank 0 writes global metadata attributes to the file
  if (metadata.rank == 0) {

    // Create a Header group and write simulation attributes
    hdf5.createGroup("Header");
    hdf5.writeAttribute<int>("Header", "NGridPoint", grid->n_grid_points);
    hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", metadata.cdim);
    hdf5.writeAttribute<double[3]>("Header", "BoxSize", metadata.dim);
    hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                                metadata.max_kernel_radius);
    hdf5.writeAttribute<double>("Header", "Redshift", metadata.redshift);

    // Create a Grids group to store simulation data for multiple kernels
    hdf5.createGroup("Grids");
  }

  // Initialize a vector to hold the local grid point counts for each cell on
  // this rank
  std::vector<int> rank_grid_point_counts(sim->nr_cells, 0);
  for (int cid = 0; cid < sim->nr_cells; cid++) {
    // Only count cells that are owned by the current rank
    if (cells[cid]->rank == metadata.rank) {
      rank_grid_point_counts[cid] = cells[cid]->grid_points.size();
    }
  }

  // Aggregate the grid point counts across all ranks using MPI_Allreduce
  // Each rank ends up with a complete view of grid_point_counts across all
  // cells
  std::vector<int> grid_point_counts(sim->nr_cells, 0);
  MPI_Allreduce(rank_grid_point_counts.data(), grid_point_counts.data(),
                sim->nr_cells, MPI_INT, MPI_SUM, comm);

  // Calculate starting indices for each cell in the global grid point array
  std::vector<int> grid_point_start(sim->nr_cells, 0);
  for (int cid = 1; cid < sim->nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Only rank 0 writes the lookup table to map cells to grid points
  if (metadata.rank == 0) {
    std::array<hsize_t, 1> cell_dims = {static_cast<hsize_t>(sim->nr_cells)};
    hdf5.createGroup("Cells");
    hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                              cell_dims);
    hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                              cell_dims);

    // Define dimensions for grid point positions dataset
    std::array<hsize_t, 2> grid_point_positions_dims = {
        static_cast<hsize_t>(grid->n_grid_points), static_cast<hsize_t>(3)};
    hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                  grid_point_positions_dims);
  }

  // Track if grid point positions have been written to avoid redundancy
  bool written_positions = false;

  // Iterate over kernel radii to write grid data for each kernel
  for (double kernel_rad : grid->kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Rank 0 creates the group for each kernel in the Grids group
    if (metadata.rank == 0) {
      hdf5.createGroup("Grids/" + kernel_name);
    }

    // Rank 0 also defines the dataset for storing grid point over-densities
    if (metadata.rank == 0) {
      std::array<hsize_t, 1> grid_point_overdens_dims = {
          static_cast<hsize_t>(grid->n_grid_points)};
      hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                        "/GridPointOverDensities",
                                    grid_point_overdens_dims);
    }

    // Synchronize all ranks to ensure consistent data access
    MPI_Barrier(comm);

// Process cells and write grid data in parallel using OpenMP
#pragma omp parallel for
    for (int cid = 0; cid < sim->nr_cells; cid++) {
      // Skip cells that are not owned by this rank or are empty
      if (cells[cid]->rank != metadata.rank || cells[cid]->grid_points.empty())
        continue;

      // Get the start index and count of grid points for this cell
      int start = grid_point_start[cid];
      int count = grid_point_counts[cid];

      // Initialize data buffers for over-densities and positions
      std::vector<double> cell_grid_ovdens(count, 0.0);
      std::vector<double> cell_grid_pos(count * 3, 0.0);

      // Fill over-density and position buffers with data from grid points
      for (size_t i = 0; i < cells[cid]->grid_points.size(); ++i) {
        const std::shared_ptr<GridPoint> &gp = cells[cid]->grid_points[i];
        cell_grid_ovdens[i] = gp->getOverDensity(kernel_rad);

        // Populate position data if positions haven't been written yet
        if (!written_positions) {
          cell_grid_pos[i * 3] = gp->loc[0];
          cell_grid_pos[i * 3 + 1] = gp->loc[1];
          cell_grid_pos[i * 3 + 2] = gp->loc[2];
        }
      }

      // Write over-density data for this cell’s grid points as a slice
      hdf5.writeDatasetSlice<double, 1>(
          "Grids/" + kernel_name + "/GridPointOverDensities", cell_grid_ovdens,
          {static_cast<hsize_t>(start)}, {static_cast<hsize_t>(count)});

      // Write grid point positions if not yet written
      if (!written_positions) {
        hdf5.writeDatasetSlice<double, 2>(
            "Grids/GridPointPositions", cell_grid_pos,
            {static_cast<hsize_t>(start), 0}, {static_cast<hsize_t>(count), 3});
      }
    } // End of OpenMP parallel for loop over cells

    // Mark positions as written after the first kernel loop
    written_positions = true;

  } // End of loop over kernel radii

  // Close the HDF5 file
  hdf5.close();
}
#endif
