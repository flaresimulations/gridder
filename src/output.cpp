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

  tic();

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata->output_file;

  message("Writing grid data to %s (serial mode)", filename.c_str());

  // Unpack the cells
  std::vector<Cell> &cells = sim->cells;

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC);
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

    // Create masses dataset if requested
    if (metadata->write_masses) {
      if (!hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                             "/GridPointMasses",
                                         grid_point_overdens_dims)) {
        error("Failed to create GridPointMasses dataset for kernel %f",
              kernel_rad);
        continue;
      }
    }

    // Process each cell
    for (size_t cid = 0; cid < sim->nr_cells; cid++) {
      Cell *cell = &cells[cid];

      // Skip empty cells
      if (cell->grid_points.size() == 0) {
        continue;
      }

      const int start_idx = grid_point_start[cid];
      const int count = grid_point_counts[cid];

      // Prepare data arrays
      std::vector<double> cell_grid_overdens;
      std::vector<double> cell_grid_masses;
      std::vector<double> cell_grid_pos;
      cell_grid_overdens.reserve(count);
      cell_grid_pos.reserve(count * 3);
      if (metadata->write_masses) {
        cell_grid_masses.reserve(count);
      }

      // Extract data from grid points
      for (const GridPoint *gp : cell->grid_points) {
        // Get overdensity for this kernel
        cell_grid_overdens.push_back(gp->getOverDensity(kernel_rad, sim));

        // Get masses if requested
        if (metadata->write_masses) {
          cell_grid_masses.push_back(gp->getMass(kernel_rad));
        }

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

      // Write masses slice if requested
      if (metadata->write_masses && !cell_grid_masses.empty()) {
        if (!hdf5.writeDatasetSlice<double, 1>(
                "Grids/" + kernel_name + "/GridPointMasses", cell_grid_masses,
                {static_cast<hsize_t>(start_idx)},
                {static_cast<hsize_t>(count)})) {
          error("Failed to write masses slice for cell %d, kernel %f", cid,
                kernel_rad);
        }
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
  toc("Writing output (in serial)");
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

  tic();

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Get the base output filepath
  const std::string base_filename = metadata->output_file;

  // Create rank-specific filename
  std::string rank_filename = base_filename;
  size_t ext_pos = rank_filename.find_last_of(".");
  if (ext_pos != std::string::npos) {
    rank_filename = rank_filename.substr(0, ext_pos) + "_rank" +
                    std::to_string(metadata->rank) +
                    rank_filename.substr(ext_pos);
  } else {
    rank_filename += "_rank" + std::to_string(metadata->rank);
  }

  message("Writing grid data to %s (parallel mode)", rank_filename.c_str());

  // Unpack the cells
  std::vector<Cell> &cells = sim->cells;

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

  // Count grid points per local cell only
  std::vector<int> local_grid_point_counts;
  std::vector<int> local_cell_ids;
  for (int cid = first_local_cell; cid < last_local_cell; cid++) {
    int count = static_cast<int>(cells[cid].grid_points.size());
    if (count > 0) {
      local_grid_point_counts.push_back(count);
      local_cell_ids.push_back(cid);
    }
  }

  // Calculate total number of local grid points
  int total_local_grid_points = 0;
  for (int count : local_grid_point_counts) {
    total_local_grid_points += count;
  }

  // Create rank-specific HDF5 file (always serial mode)
  HDF5Helper hdf5(rank_filename, H5F_ACC_TRUNC);
  if (!hdf5.isOpen()) {
    error("Rank %d: Failed to create HDF5 file: %s", metadata->rank,
          rank_filename.c_str());
    return;
  }

  // Create groups
  if (!hdf5.createGroup("Header")) {
    error("Rank %d: Failed to create Header group", metadata->rank);
    return;
  }

  if (!hdf5.createGroup("Grids")) {
    error("Rank %d: Failed to create Grids group", metadata->rank);
    return;
  }

  if (!hdf5.createGroup("Cells")) {
    error("Rank %d: Failed to create Cells group", metadata->rank);
    return;
  }

  // Write header attributes (each rank writes its own copy)
  hdf5.writeAttribute<int>("Header", "MPI_Rank", metadata->rank);
  hdf5.writeAttribute<int>("Header", "MPI_Size", metadata->size);
  hdf5.writeAttribute<int>("Header", "LocalGridPoints",
                           total_local_grid_points);
  hdf5.writeAttribute<int>("Header", "LocalCells", nr_local_cells);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              grid->max_kernel_radius);

  // Write local cell information
  if (!local_cell_ids.empty()) {
    std::array<hsize_t, 1> local_cell_dims = {
        static_cast<hsize_t>(local_cell_ids.size())};

    if (!hdf5.writeDataset<int, 1>("Cells/LocalCellIDs", local_cell_ids,
                                   local_cell_dims)) {
      error("Rank %d: Failed to write LocalCellIDs dataset", metadata->rank);
      return;
    }

    if (!hdf5.writeDataset<int, 1>("Cells/LocalGridPointCounts",
                                   local_grid_point_counts, local_cell_dims)) {
      error("Rank %d: Failed to write LocalGridPointCounts dataset",
            metadata->rank);
      return;
    }
  }

  // Write grid data if we have any
  if (total_local_grid_points > 0) {
    // Prepare local data
    std::vector<double> local_positions;
    local_positions.reserve(total_local_grid_points * 3);

    // Create overdensity arrays for each kernel
    std::vector<std::vector<double>> local_overdens(grid->kernel_radii.size());
    for (auto &overdens_vec : local_overdens) {
      overdens_vec.reserve(total_local_grid_points);
    }

    // If we are writing out masses, we need to prepare a vector for them in
    // each kernel too
    std::vector<std::vector<double>> local_masses;
    if (metadata->write_masses) {
      local_masses.resize(grid->kernel_radii.size());
      for (auto &mass_vec : local_masses) {
        mass_vec.reserve(total_local_grid_points);
      }
    }

    // Extract data from local grid points
    for (int cid = first_local_cell; cid < last_local_cell; cid++) {
      Cell *cell = &cells[cid];
      for (const GridPoint *gp : cell->grid_points) {
        // Store positions
        local_positions.push_back(gp->loc[0]);
        local_positions.push_back(gp->loc[1]);
        local_positions.push_back(gp->loc[2]);

        // Store overdensities for each kernel
        for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
          local_overdens[k].push_back(
              gp->getOverDensity(grid->kernel_radii[k], sim));
        }

        // Store masses if desired
        if (metadata->write_masses) {
          for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
            local_masses[k].push_back(gp->getMass(grid->kernel_radii[k]));
          }
        }
      }
    }

    // Write position data
    std::array<hsize_t, 2> pos_dims = {
        static_cast<hsize_t>(total_local_grid_points), 3};
    if (!hdf5.writeDataset<double, 2>("Grids/GridPointPositions",
                                      local_positions, pos_dims)) {
      error("Rank %d: Failed to write GridPointPositions dataset",
            metadata->rank);
      return;
    }

    // Write overdensity data for each kernel
    for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
      double kernel_rad = grid->kernel_radii[k];
      std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

      if (!hdf5.createGroup("Grids/" + kernel_name)) {
        error("Rank %d: Failed to create kernel group: %s", metadata->rank,
              kernel_name.c_str());
        continue;
      }

      std::array<hsize_t, 1> overdens_dims = {
          static_cast<hsize_t>(total_local_grid_points)};
      if (!hdf5.writeDataset<double, 1>("Grids/" + kernel_name +
                                            "/GridPointOverDensities",
                                        local_overdens[k], overdens_dims)) {
        error("Rank %d: Failed to write overdensity dataset for kernel %f",
              metadata->rank, kernel_rad);
      }

      // Write the masses if desired
      if (metadata->write_masses) {
        if (!hdf5.writeDataset<double, 1>("Grids/" + kernel_name +
                                              "/GridPointMasses",
                                          local_masses[k], overdens_dims)) {
          error("Rank %d: Failed to write mass dataset for kernel %f",
                metadata->rank, kernel_rad);
        }
      }
    }
  }

  // Close the rank file
  hdf5.close();

  // Synchronize all ranks before creating virtual file
  MPI_Barrier(MPI_COMM_WORLD);

  // Only rank 0 creates the virtual file
  if (metadata->rank == 0) {
    createVirtualFile(base_filename, metadata->size, sim, grid);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  toc("Writing output (in parallel)");
}

/**
 * @brief Create virtual HDF5 file that combines all rank files
 *
 * @param base_filename The main output filename
 * @param num_ranks Number of MPI ranks
 * @param sim Pointer to the simulation object
 * @param grid Pointer to the grid object
 */
void createVirtualFile(const std::string &base_filename, int num_ranks,
                       Simulation *sim, Grid *grid) {

  message("Creating virtual HDF5 file: %s", base_filename.c_str());

  // Get the metadata
  Metadata *metadata = &Metadata::getInstance();

  // Create the virtual file
  HDF5Helper hdf5(base_filename, H5F_ACC_TRUNC);
  if (!hdf5.isOpen()) {
    error("Failed to create virtual HDF5 file: %s", base_filename.c_str());
    return;
  }

  // First read all rank files to determine total grid points and offsets
  std::vector<int> rank_grid_points(num_ranks, 0);
  std::vector<int> rank_offsets(num_ranks, 0);
  int total_grid_points = 0;

  for (int rank = 0; rank < num_ranks; rank++) {
    std::string rank_filename = base_filename;
    size_t ext_pos = rank_filename.find_last_of(".");
    if (ext_pos != std::string::npos) {
      rank_filename = rank_filename.substr(0, ext_pos) + "_rank" +
                      std::to_string(rank) + rank_filename.substr(ext_pos);
    } else {
      rank_filename += "_rank" + std::to_string(rank);
    }

    // Read header from rank file to get local grid point count
    HDF5Helper rank_file(rank_filename, H5F_ACC_RDONLY);
    if (rank_file.isOpen()) {
      int local_count = 0;
      if (rank_file.readAttribute("Header", "LocalGridPoints", local_count)) {
        rank_grid_points[rank] = local_count;
        rank_offsets[rank] = total_grid_points;
        total_grid_points += local_count;
      }
      rank_file.close();
    }
  }

  // Create the main file structure
  if (!hdf5.createGroup("Header")) {
    error("Failed to create Header group in virtual file");
    return;
  }

  if (!hdf5.createGroup("Grids")) {
    error("Failed to create Grids group in virtual file");
    return;
  }

  if (!hdf5.createGroup("Cells")) {
    error("Failed to create Cells group in virtual file");
    return;
  }

  // Write global header attributes
  hdf5.writeAttribute<int>("Header", "NGridPoint", total_grid_points);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", sim->cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", sim->dim);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              grid->max_kernel_radius);
  hdf5.writeAttribute<double>("Header", "Redshift", sim->redshift);
  hdf5.writeAttribute<int>("Header", "MPI_Size", num_ranks);

  // Create virtual datasets for positions
  std::array<hsize_t, 2> global_pos_dims = {
      static_cast<hsize_t>(total_grid_points), 3};
  if (!hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                     global_pos_dims)) {
    error("Failed to create virtual GridPointPositions dataset");
    return;
  }

  // Create virtual datasets for each kernel
  for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
    double kernel_rad = grid->kernel_radii[k];
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    if (!hdf5.createGroup("Grids/" + kernel_name)) {
      error("Failed to create virtual kernel group: %s", kernel_name.c_str());
      continue;
    }

    std::array<hsize_t, 1> global_overdens_dims = {
        static_cast<hsize_t>(total_grid_points)};
    if (!hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                           "/GridPointOverDensities",
                                       global_overdens_dims)) {
      error("Failed to create virtual overdensity dataset for kernel %f",
            kernel_rad);
    }

    // Create masses dataset if requested
    if (metadata->write_masses) {
      if (!hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                             "/GridPointMasses",
                                         global_overdens_dims)) {
        error("Failed to create virtual mass dataset for kernel %f",
              kernel_rad);
      }
    }
  }

  // Now populate the virtual datasets by copying data from rank files
  for (int rank = 0; rank < num_ranks; rank++) {
    if (rank_grid_points[rank] == 0)
      continue; // Skip ranks with no data

    std::string rank_filename = base_filename;
    size_t ext_pos = rank_filename.find_last_of(".");
    if (ext_pos != std::string::npos) {
      rank_filename = rank_filename.substr(0, ext_pos) + "_rank" +
                      std::to_string(rank) + rank_filename.substr(ext_pos);
    } else {
      rank_filename += "_rank" + std::to_string(rank);
    }

    HDF5Helper rank_file(rank_filename, H5F_ACC_RDONLY);
    if (!rank_file.isOpen()) {
      error("Failed to open rank file for reading: %s", rank_filename.c_str());
      continue;
    }

    // Read and write position data
    std::vector<double> rank_positions;
    if (rank_file.readDataset("Grids/GridPointPositions", rank_positions)) {
      if (!hdf5.writeDatasetSlice<double, 2>(
              "Grids/GridPointPositions", rank_positions,
              {static_cast<hsize_t>(rank_offsets[rank]), 0},
              {static_cast<hsize_t>(rank_grid_points[rank]), 3})) {
        error("Failed to write position slice for rank %d", rank);
      }
    }

    // Read and write overdensity data for each kernel
    for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
      double kernel_rad = grid->kernel_radii[k];
      std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

      std::vector<double> rank_overdens;
      if (rank_file.readDataset("Grids/" + kernel_name +
                                    "/GridPointOverDensities",
                                rank_overdens)) {
        if (!hdf5.writeDatasetSlice<double, 1>(
                "Grids/" + kernel_name + "/GridPointOverDensities",
                rank_overdens, {static_cast<hsize_t>(rank_offsets[rank])},
                {static_cast<hsize_t>(rank_grid_points[rank])})) {
          error("Failed to write overdensity slice for rank %d, kernel %f",
                rank, kernel_rad);
        }
      }
    }

    // Read and write the masses if requested
    if (metadata->write_masses) {
      for (size_t k = 0; k < grid->kernel_radii.size(); k++) {
        double kernel_rad = grid->kernel_radii[k];
        std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

        std::vector<double> rank_masses;
        if (rank_file.readDataset("Grids/" + kernel_name + "/GridPointMasses",
                                  rank_masses)) {
          if (!hdf5.writeDatasetSlice<double, 1>(
                  "Grids/" + kernel_name + "/GridPointMasses", rank_masses,
                  {static_cast<hsize_t>(rank_offsets[rank])},
                  {static_cast<hsize_t>(rank_grid_points[rank])})) {
            error("Failed to write mass slice for rank %d, kernel %f", rank,
                  kernel_rad);
          }
        }
      }
    }

    rank_file.close();
  }

  // Create cell lookup information for the combined data
  std::vector<int> global_grid_point_counts(sim->nr_cells, 0);
  std::vector<int> global_grid_point_start(sim->nr_cells, 0);

  // Build global cell information from all ranks
  int current_offset = 0;
  for (size_t cid = 0; cid < sim->nr_cells; cid++) {
    global_grid_point_start[cid] = current_offset;
    global_grid_point_counts[cid] =
        static_cast<int>(sim->cells[cid].grid_points.size());
    current_offset += global_grid_point_counts[cid];
  }

  // Write global cell datasets
  std::array<hsize_t, 1> cell_dims = {static_cast<hsize_t>(sim->nr_cells)};
  hdf5.writeDataset<int, 1>("Cells/GridPointStart", global_grid_point_start,
                            cell_dims);
  hdf5.writeDataset<int, 1>("Cells/GridPointCounts", global_grid_point_counts,
                            cell_dims);

  hdf5.close();
  message(
      "Successfully created virtual HDF5 file @ %s with %d total grid points ",
      base_filename.c_str(), total_grid_points);
}
#endif
