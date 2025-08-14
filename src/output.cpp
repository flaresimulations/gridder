/**
 * @file output.cpp
 * @brief Implementation of grid output functions with proper serial/parallel
 * HDF5 support
 */

// Standard includes
#include <algorithm>
#include <numeric>

// Local includes
#include "grid_point.hpp"
#include "hdf_io.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "simulation.hpp"

/**
 * @brief Write grid data to HDF5 file in serial mode
 *
 * This function writes all grid point data to a single HDF5 file using
 * serial I/O operations. It organizes data by cells and kernel radii.
 *
 * @param sim Pointer to the simulation object
 * @param grid Pointer to the grid object containing grid points and kernel data
 */
void writeGridFileSerial(Simulation *sim, Grid *grid) {
  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata->output_file;

  message("Writing grid data to %s (serial mode)", filename.c_str());

  // Unpack the cells
  std::vector<Cell>& cells = sim->cells;

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC,
                  false); // No collective I/O for serial
  if (!hdf5.isOpen()) {
    error("Failed to create HDF5 file for serial output");
    return;
  }

  // Create the Header group and write out the metadata
  if (!hdf5.createGroup("Header")) {
    error("Failed to create Header group");
    return;
  }

  // Write header attributes
  hdf5.writeAttribute<int>("Header", "NGridPoint", grid->n_grid_points);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", sim->cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", sim->dim);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              grid->max_kernel_radius);
  hdf5.writeAttribute<double>("Header", "Redshift", sim->redshift);

  // Create the Grids group
  if (!hdf5.createGroup("Grids")) {
    error("Failed to create Grids group");
    return;
  }

  // Loop over cells and collect grid point counts
  std::vector<int> grid_point_counts(sim->nr_cells, 0);
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    grid_point_counts[cid] = static_cast<int>(cells[cid].grid_points.size());
  }

  // Convert counts to start indices for cell lookup
  std::vector<int> grid_point_start(sim->nr_cells, 0);
  for (size_t cid = 1; cid < sim->nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Create and write cell lookup tables
  if (!hdf5.createGroup("Cells")) {
    error("Failed to create Cells group");
    return;
  }

  std::array<hsize_t, 1> sim_cell_dims = {static_cast<hsize_t>(sim->nr_cells)};
  if (!hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                                 sim_cell_dims)) {
    error("Failed to write GridPointStart dataset");
    return;
  }
  if (!hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                                 sim_cell_dims)) {
    error("Failed to write GridPointCounts dataset");
    return;
  }

  // Create dataset for grid positions
  std::array<hsize_t, 2> grid_point_positions_dims = {
      static_cast<hsize_t>(grid->n_grid_points), static_cast<hsize_t>(3)};
  if (!hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                     grid_point_positions_dims)) {
    error("Failed to create GridPointPositions dataset");
    return;
  }

  // Track whether we've written positions
  bool written_positions = false;

  // Loop over kernel radii and write grid data
  for (double kernel_rad : grid->kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Create kernel group
    if (!hdf5.createGroup("Grids/" + kernel_name)) {
      error("Failed to create kernel group: %s", kernel_name.c_str());
      continue;
    }

    // Create overdensity dataset for this kernel
    std::array<hsize_t, 1> grid_point_overdens_dims = {
        static_cast<hsize_t>(grid->n_grid_points)};
    if (!hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                           "/GridPointOverDensities",
                                       grid_point_overdens_dims)) {
      error("Failed to create GridPointOverDensities dataset for kernel %f",
            kernel_rad);
      continue;
    }

    // Process each cell
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      Cell* cell = &cells[cid];

      // Skip empty cells
      if (cell->grid_points.size() == 0) {
        continue;
      }

      const int start_idx = grid_point_start[cid];
      const int count = grid_point_counts[cid];

      // Prepare data arrays
      std::vector<double> cell_grid_overdens;
      std::vector<double> cell_grid_pos;
      cell_grid_overdens.reserve(count);
      cell_grid_pos.reserve(count * 3);

      // Extract data from grid points
      for (const GridPoint* gp : cell->grid_points) {
        // Get overdensity for this kernel
        cell_grid_overdens.push_back(gp->getOverDensity(kernel_rad, sim));

        // Store positions if not done yet
        if (!written_positions) {
          cell_grid_pos.push_back(gp->loc[0]);
          cell_grid_pos.push_back(gp->loc[1]);
          cell_grid_pos.push_back(gp->loc[2]);
        }
      }

      // Write overdensity slice
      if (!hdf5.writeDatasetSlice<double, 1>(
              "Grids/" + kernel_name + "/GridPointOverDensities",
              cell_grid_overdens, {static_cast<hsize_t>(start_idx)},
              {static_cast<hsize_t>(count)})) {
        error("Failed to write overdensity slice for cell %d, kernel %f", cid,
              kernel_rad);
        continue;
      }

      // Write position slice if needed
      if (!written_positions && !cell_grid_pos.empty()) {
        if (!hdf5.writeDatasetSlice<double, 2>(
                "Grids/GridPointPositions", cell_grid_pos,
                {static_cast<hsize_t>(start_idx), 0},
                {static_cast<hsize_t>(count), 3})) {
          error("Failed to write position slice for cell %d", cid);
        }
      }
    } // End cell loop

    // Mark positions as written after first kernel
    written_positions = true;

  } // End kernel loop

  // Close the HDF5 file
  hdf5.close();
  message("Successfully wrote grid data (serial mode)");
}

#ifdef WITH_MPI
/**
 * @brief Write grid data to HDF5 file in parallel mode
 *
 * This function coordinates between MPI ranks to write grid data using
 * proper parallel HDF5 operations. Each rank writes its local data to
 * appropriate hyperslabs in the shared datasets.
 *
 * @param sim Pointer to the simulation object
 * @param grid Pointer to the grid object containing grid points and kernel data
 */
void writeGridFileParallel(Simulation *sim, Grid *grid) {
  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata->output_file;

  if (metadata->rank == 0) {
    message("Writing grid data to %s (parallel mode with %d ranks)",
            filename.c_str(), metadata->size);
  }

  // Unpack the cells
  std::vector<Cell>& cells = sim->cells;

  // Calculate local cell range for this rank
  int nr_cells_before = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    if (cells[cid].rank == metadata->rank) {
      break;
    }
    nr_cells_before++;
  }

  const int first_local_cell = nr_cells_before;
  const int nr_local_cells = metadata->nr_local_cells;
  const int last_local_cell = first_local_cell + nr_local_cells;

  // Count grid points per cell (only for local cells)
  std::vector<int> local_grid_point_counts(sim->nr_cells, 0);
  for (int cid = first_local_cell; cid < last_local_cell; cid++) {
    local_grid_point_counts[cid] =
        static_cast<int>(cells[cid].grid_points.size());
  }

  // Gather grid point counts from all ranks
  std::vector<int> grid_point_counts(sim->nr_cells, 0);
  MPI_Allreduce(local_grid_point_counts.data(), grid_point_counts.data(),
                sim->nr_cells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Convert counts to start indices
  std::vector<int> grid_point_start(sim->nr_cells, 0);
  for (size_t cid = 1; cid < sim->nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Calculate total number of local grid points
  int total_local_grid_points = 0;
  for (int cid = first_local_cell; cid < last_local_cell; cid++) {
    total_local_grid_points += grid_point_counts[cid];
  }

  // Calculate global offset for this rank's data
  std::vector<int> local_totals(metadata->size, 0);
  MPI_Allgather(&total_local_grid_points, 1, MPI_INT, local_totals.data(), 1,
                MPI_INT, MPI_COMM_WORLD);

  int global_offset = 0;
  for (int r = 0; r < metadata->rank; r++) {
    global_offset += local_totals[r];
  }

  // Create HDF5 file with parallel access
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC, true); // Use collective I/O
  if (!hdf5.isOpen()) {
    error("Rank %d: Failed to create HDF5 file for parallel output",
          metadata->rank);
    return;
  }

  // Only rank 0 creates groups and writes metadata
  if (metadata->rank == 0) {
    // Create header group and write metadata
    if (!hdf5.createGroup("Header")) {
      error("Rank 0: Failed to create Header group");
      return;
    }

    hdf5.writeAttribute<int>("Header", "NGridPoint", grid->n_grid_points);
    hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", sim->cdim);
    hdf5.writeAttribute<double[3]>("Header", "BoxSize", sim->dim);
    hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                                grid->max_kernel_radius);
    hdf5.writeAttribute<double>("Header", "Redshift", sim->redshift);

    // Create groups
    hdf5.createGroup("Grids");
    hdf5.createGroup("Cells");

    // Write cell lookup tables
    std::array<hsize_t, 1> sim_cell_dims = {
        static_cast<hsize_t>(sim->nr_cells)};
    hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                              sim_cell_dims);
    hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                              sim_cell_dims);

    // Create grid position dataset
    std::array<hsize_t, 2> grid_point_positions_dims = {
        static_cast<hsize_t>(grid->n_grid_points), static_cast<hsize_t>(3)};
    hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                  grid_point_positions_dims);

    // Create datasets for each kernel
    for (double kernel_rad : grid->kernel_radii) {
      std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);
      hdf5.createGroup("Grids/" + kernel_name);

      std::array<hsize_t, 1> grid_point_overdens_dims = {
          static_cast<hsize_t>(grid->n_grid_points)};
      hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                        "/GridPointOverDensities",
                                    grid_point_overdens_dims);
    }
  }

  // Synchronize all ranks after setup
  MPI_Barrier(MPI_COMM_WORLD);

  // Prepare local data for writing
  if (total_local_grid_points > 0) {
    std::vector<double> local_positions;
    local_positions.reserve(total_local_grid_points * 3);

    // Create overdensity arrays for each kernel
    std::vector<std::vector<double>> local_overdens(grid->kernel_radii.size());
    for (auto &overdens_vec : local_overdens) {
      overdens_vec.reserve(total_local_grid_points);
    }

    // Extract data from local grid points
    for (int cid = first_local_cell; cid < last_local_cell; cid++) {
      Cell* cell = &cells[cid];

      for (const GridPoint* gp : cell->grid_points) {
        // Store positions
        local_positions.push_back(gp->loc[0]);
        local_positions.push_back(gp->loc[1]);
        local_positions.push_back(gp->loc[2]);

        // Store overdensities for each kernel
        for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
          local_overdens[k].push_back(
              gp->getOverDensity(grid->kernel_radii[k], sim));
        }
      }
    }

    // Write position data
    if (!local_positions.empty()) {
      if (!hdf5.writeDatasetSlice<double, 2>(
              "Grids/GridPointPositions", local_positions,
              {static_cast<hsize_t>(global_offset), 0},
              {static_cast<hsize_t>(total_local_grid_points), 3})) {
        error("Rank %d: Failed to write position data", metadata->rank);
      }
    }

    // Write overdensity data for each kernel
    for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
      if (!local_overdens[k].empty()) {
        std::string kernel_name =
            "Kernel_" + std::to_string(grid->kernel_radii[k]);
        if (!hdf5.writeDatasetSlice<double, 1>(
                "Grids/" + kernel_name + "/GridPointOverDensities",
                local_overdens[k], {static_cast<hsize_t>(global_offset)},
                {static_cast<hsize_t>(total_local_grid_points)})) {
          error("Rank %d: Failed to write overdensity data for kernel %f",
                metadata->rank, grid->kernel_radii[k]);
        }
      }
    }
  }

  // Synchronize before closing
  MPI_Barrier(MPI_COMM_WORLD);

  // Close the file
  hdf5.close();

  if (metadata->rank == 0) {
    message("Successfully wrote grid data (parallel mode)");
  }
}
#endif
