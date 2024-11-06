/**
 * @file hdf_io.hpp
 * @brief Header file for the HDF5 I/O helper class
 */
#ifndef HDF_IO_H_
#define HDF_IO_H_

// Standard includes
#include <array>
#include <hdf5.h> // HDF5 C API
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "logger.hpp"

class HDF5Helper {
public:
  hid_t file_id; ///< HDF5 file identifier

#ifdef WITH_MPI
  MPI_Comm comm; ///< MPI communicator
  MPI_Info info; ///< MPI info object for additional options
#endif

#ifdef WITH_MPI
  /**
   * @brief Constructor for a HDF5 helper object with parallel I/O
   *
   * @param filename The name of the HDF5 file to open.
   * @param accessMode The file access mode, e.g., H5F_ACC_RDONLY, H5F_ACC_RDWR.
   * @param communicator The MPI communicator to use (default is
   * MPI_COMM_WORLD).
   * @param file_info Optional MPI_Info object for additional hints (default is
   * MPI_INFO_NULL).
   */
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY,
             MPI_Comm communicator = MPI_COMM_WORLD,
             MPI_Info file_info = MPI_INFO_NULL)
      : comm(communicator), info(file_info), file_open(true),
        file_closed(false) {
    // Set up the HDF5 file access property list for parallel I/O
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, communicator, file_info);

    // Open or create the HDF5 file in parallel mode
    if (accessMode == H5F_ACC_RDONLY) {
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
    } else {
      file_id = H5Fcreate(filename.c_str(), accessMode, H5P_DEFAULT, fapl);
    }

    if (file_id < 0)
      error("Failed to open HDF5 file: %s", filename.c_str());

    H5Pclose(fapl); // Close the property list after use
  }
#else
  /**
   * @brief Constructor for a HDF5 helper object with serial I/O
   *
   * @param filename The name of the HDF5 file to open.
   * @param accessMode The file access mode, e.g., H5F_ACC_RDONLY, H5F_ACC_RDWR.
   */
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY)
      : file_open(true), file_closed(false) {
    if (accessMode == H5F_ACC_RDONLY) {
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    } else {
      file_id =
          H5Fcreate(filename.c_str(), accessMode, H5P_DEFAULT, H5P_DEFAULT);
    }

    if (file_id < 0)
      error("Failed to open HDF5 file: %s", filename.c_str());
  }
#endif // WITH_MPI

  /**
   * @brief Destructor closes the file if it is still open
   */
  ~HDF5Helper() {
    if (!file_closed)
      H5Fclose(file_id);
  }

  /**
   * @brief Closes the HDF5 file, marking it as closed
   */
  void close() {
    if (!file_closed) {
      H5Fclose(file_id);
      file_closed = true;
      file_open = false;
    }
  }

  /**
   * @brief Creates a group within the HDF5 file
   *
   * @param groupName The name of the group to create
   * @return true if the group was created successfully, false otherwise
   */
  bool createGroup(const std::string &groupName) {
    hid_t group_id = H5Gcreate(file_id, groupName.c_str(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0)
      return false;

    H5Gclose(group_id);
    return true;
  }

  /**
   * @brief Opens an existing group within the HDF5 file
   *
   * @param groupName The name of the group to open
   * @return The group identifier if opened successfully, or -1 on failure
   */
  hid_t openGroup(const std::string &groupName) {
    hid_t group_id = H5Gopen(file_id, groupName.c_str(), H5P_DEFAULT);
    if (group_id < 0)
      return -1;

    H5Gclose(group_id);
    return group_id;
  }

  /**
   * @brief Reads an attribute from a specified HDF5 object
   *
   * Reads the attribute data and stores it in the provided variable.
   *
   * @tparam T The data type of the attribute (supports scalar and fixed-size
   * arrays)
   * @param objName Name of the HDF5 object from which to read the attribute
   * @param attributeName Name of the attribute to read
   * @param attributeValue Reference to store the read attribute data
   * @return true if the attribute was read successfully, false otherwise
   */
  template <typename T>
  bool readAttribute(const std::string &objName,
                     const std::string &attributeName, T &attributeValue) {
    hid_t obj_id = H5Oopen(file_id, objName.c_str(), H5P_DEFAULT);
    if (obj_id < 0)
      return false;

    hid_t attr_id = H5Aopen(obj_id, attributeName.c_str(), H5P_DEFAULT);
    hid_t attr_type = H5Aget_type(attr_id);

    herr_t status = H5Aread(attr_id, attr_type, &attributeValue);

    H5Aclose(attr_id);
    H5Oclose(obj_id);

    return status >= 0;
  }

  /**
   * @brief Writes an attribute to a specified HDF5 object
   *
   * Creates an attribute on an HDF5 object (e.g., group or dataset) and writes
   * either a single value or an array to it, inferred from the input data.
   *
   * @tparam T The data type of the attribute (supports scalar and fixed-size
   * arrays)
   * @param objName Name of the HDF5 object to attach the attribute to
   * @param attributeName Name of the attribute to create
   * @param attributeValue The value to write as the attribute
   * @return true if the attribute was written successfully, false otherwise
   */
  template <typename T>
  bool writeAttribute(const std::string &objName,
                      const std::string &attributeName,
                      const T &attributeValue) {
    hid_t obj_id = H5Oopen(file_id, objName.c_str(), H5P_DEFAULT);
    if (obj_id < 0)
      return false;

    hid_t attr_type = getHDF5Type<T>();
    hid_t dataspace_id;

    if constexpr (std::is_array_v<T>) {
      hsize_t dims[1] = {std::extent_v<T>};
      dataspace_id = H5Screate_simple(1, dims, nullptr);
    } else {
      dataspace_id = H5Screate(H5S_SCALAR);
    }

    hid_t attr_id = H5Acreate(obj_id, attributeName.c_str(), attr_type,
                              dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
      H5Oclose(obj_id);
      H5Sclose(dataspace_id);
      return false;
    }

    herr_t status = H5Awrite(attr_id, attr_type, &attributeValue);
    H5Aclose(attr_id);
    H5Sclose(dataspace_id);
    H5Oclose(obj_id);

    return status >= 0;
  }

  /**
   * @brief Reads a complete dataset from the file in serial mode (avoiding
   * collective I/O)
   *
   * This function opens an existing dataset and reads its contents into a
   * provided vector. It supports reading multi-dimensional datasets and
   * automatically adjusts the buffer size based on the dataset dimensions.
   *
   * @tparam T Data type of the dataset elements
   * @param datasetName Name of the dataset to read
   * @param data Reference to a vector where the data will be stored
   * @return true if the dataset was read successfully, false otherwise
   */
  template <typename T>
  bool readDataset(const std::string &datasetName, std::vector<T> &data) {
    // Open the dataset
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    // Get the dataspace and determine the number of elements
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    std::vector<hsize_t> dims(rank);
    H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
    hsize_t num_elements = 1;
    for (const auto &dim : dims)
      num_elements *= dim;

    // Resize the buffer to hold the entire dataset
    data.resize(num_elements);

    // Read the data (non-collective, serial mode)
    herr_t status = H5Dread(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, data.data());

    // Close HDF5 identifiers
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    return status >= 0;
  }

  /**
   * @brief Creates a new dataset in the HDF5 file
   *
   * This function creates a dataset with the specified name and dimensions.
   * It initializes the dataset with collective I/O settings to ensure efficient
   * access in parallel environments.
   *
   * @tparam T Data type of the dataset elements
   * @tparam Rank Rank (number of dimensions) of the dataset
   * @param datasetName Name of the dataset to create
   * @param dims Array specifying the size of each dimension
   * @return true if the dataset was created successfully, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool createDataset(const std::string &datasetName,
                     const std::array<hsize_t, Rank> &dims) {
    // Create the dataspace for the dataset with the specified dimensions
    hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), nullptr);
    if (dataspace_id < 0)
      return false;

    // Define properties for chunking and collective I/O
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, Rank, dims.data());

    hid_t dataset_id =
        H5Dcreate(file_id, (datasetName).c_str(), getHDF5Type<T>(),
                  dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Pclose(plist_id);
    H5Sclose(dataspace_id);

    if (dataset_id < 0)
      return false;

    // Close the dataset identifier
    H5Dclose(dataset_id);
    return true;
  }

  /**
   * @brief Writes a complete dataset to the file
   *
   * This function creates a dataset with the specified name and dimensions,
   * then writes data to it.
   *
   * @tparam T Data type of the dataset
   * @tparam Rank Rank (number of dimensions) of the dataset
   * @param datasetName Name of the dataset to create
   * @param data The data to write to the dataset
   * @param dims Array specifying the size of each dimension
   * @return true if the dataset was written successfully, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool writeDataset(const std::string &datasetName, const std::vector<T> &data,
                    const std::array<hsize_t, Rank> &dims) {
    hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), nullptr);
    if (dataspace_id < 0)
      return false;

    hid_t dataset_id =
        H5Dcreate(file_id, datasetName.c_str(), getHDF5Type<T>(), dataspace_id,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
      H5Sclose(dataspace_id);
      return false;
    }

    herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return status >= 0;
  }

  /**
   * @brief Writes a slice of data to an existing dataset
   *
   * Writes a slice of data to a specific hyperslab within an existing dataset,
   * allowing parallel processes to write to separate portions.
   *
   * @tparam T Data type of the dataset
   * @tparam Rank Rank (number of dimensions) of the dataset
   * @param datasetName Name of the dataset to write to
   * @param data The data to write to the hyperslab
   * @param start Array specifying the starting indices of the hyperslab
   * @param count Array specifying the size of the hyperslab
   * @return true if the data slice was written successfully, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool writeDatasetSlice(const std::string &datasetName,
                         const std::vector<T> &data,
                         const std::array<hsize_t, Rank> &start,
                         const std::array<hsize_t, Rank> &count) {
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    hid_t filespace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start.data(), nullptr,
                        count.data(), nullptr);

    hid_t memspace_id = H5Screate_simple(Rank, count.data(), nullptr);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), memspace_id,
                             filespace_id, plist_id, data.data());
    H5Pclose(plist_id);
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);

    return status >= 0;
  }

  /**
   * @brief Checks if the given dataset is a virtual dataset (VDS)
   *
   * @param datasetName The name of the dataset to check
   * @return true if the dataset is virtual, false otherwise
   */
  bool isVirtualDataset(const std::string &datasetName) {
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    hid_t dapl_id = H5Dget_create_plist(dataset_id);
    bool is_virtual = (H5Pget_layout(dapl_id) == H5D_VIRTUAL);

    H5Pclose(dapl_id);
    H5Dclose(dataset_id);
    return is_virtual;
  }

  /**
   * @brief Reads a slice from a virtual dataset by accessing its source
   * datasets
   *
   * If the dataset is virtual, this method identifies the source datasets and
   * reads the requested slice directly from them. Otherwise, it falls back to
   * standard slice reading.
   *
   * @tparam T Data type of the dataset
   * @tparam Rank Rank (number of dimensions) of the dataset
   * @param datasetName Name of the dataset to read from
   * @param data Vector to store the slice data
   * @param start_array Start indices for the slice
   * @param count_array Size of the slice in each dimension
   * @return true if the slice was read successfully, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool readDatasetSliceFromVDS(const std::string &datasetName,
                               std::vector<T> &data,
                               const std::array<hsize_t, Rank> &start_array,
                               const std::array<hsize_t, Rank> &count_array) {

    // Open the virtual dataset and get its creation property list
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    hid_t dapl_id = H5Dget_create_plist(dataset_id);

    // Get the count of source datasets in the virtual dataset
    size_t src_count;
    H5Pget_virtual_count(dapl_id, &src_count);

    for (size_t i = 0; i < src_count; i++) {
      char src_file[128];
      char src_dset[128];

      // Get the source file and dataset names
      H5Pget_virtual_filename(dapl_id, i, src_file, sizeof(src_file));
      H5Pget_virtual_dsetname(dapl_id, i, src_dset, sizeof(src_dset));

      // Open each source dataset for reading
      hid_t src_file_id = H5Fopen(src_file, H5F_ACC_RDONLY, H5P_DEFAULT);
      hid_t src_dset_id = H5Dopen(src_file_id, src_dset, H5P_DEFAULT);

      // Determine the slice within this source dataset
      hid_t src_space = H5Dget_space(src_dset_id);
      hsize_t src_offset[Rank];
      hsize_t src_count[Rank];

      // Here, map `start_array` and `count_array` to the source dataset layout
      // You may need to adjust based on the virtual dataset mapping

      // Select the hyperslab in the source dataset
      H5Sselect_hyperslab(src_space, H5S_SELECT_SET, src_offset, nullptr,
                          src_count, nullptr);

      // Define the memory dataspace and read data
      hid_t mem_space = H5Screate_simple(Rank, src_count, nullptr);
      data.resize(std::accumulate(src_count, src_count + Rank, 1,
                                  std::multiplies<hsize_t>()));
      herr_t status = H5Dread(src_dset_id, getHDF5Type<T>(), mem_space,
                              src_space, H5P_DEFAULT, data.data());

      // Close resources for this source dataset
      H5Sclose(src_space);
      H5Dclose(src_dset_id);
      H5Fclose(src_file_id);

      if (status < 0)
        return false;
    }

    // Close dataset and property list
    H5Pclose(dapl_id);
    H5Dclose(dataset_id);
    return true;
  }

  /**
   * @brief Reads a dataset slice, determining if the dataset is virtual and, if
   * so, using the source datasets.
   *
   * @tparam T Data type of the dataset
   * @tparam Rank Rank (number of dimensions) of the dataset
   * @param datasetName Name of the dataset to read from
   * @param data Vector to store the slice data
   * @param start_array Start indices for the slice
   * @param count_array Size of the slice in each dimension
   * @return true if the slice was read successfully, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool readDatasetSlice(const std::string &datasetName, std::vector<T> &data,
                        const std::array<hsize_t, Rank> &start_array,
                        const std::array<hsize_t, Rank> &count_array) {

    /* If we have a virtual dataset, read the slice from the source datasets */
    if (isVirtualDataset(datasetName)) {
      return readDatasetSliceFromVDS(datasetName, data, start_array,
                                     count_array);
    }

    // Open the dataset and retrieve its dataspace
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    hid_t dataspace = H5Dget_space(dataset_id);

    // Get the rank and dimensions of the dataset
    int dataset_rank = H5Sget_simple_extent_ndims(dataspace);
    if (dataset_rank != Rank) {
      error("Dataset rank (%d) does not match the template rank (%zu).",
            dataset_rank, Rank);
      H5Sclose(dataspace);
      H5Dclose(dataset_id);
      return false;
    }

    std::array<hsize_t, Rank> dims;
    H5Sget_simple_extent_dims(dataspace, dims.data(), nullptr);

    // Validate that the requested slice is within bounds for each dimension
    for (std::size_t i = 0; i < Rank; ++i) {
      if (start_array[i] + count_array[i] > dims[i]) {
        error("Requested slice out of bounds in dimension %zu (start=%llu, "
              "count=%llu, dim=%llu).",
              i, static_cast<unsigned long long>(start_array[i]),
              static_cast<unsigned long long>(count_array[i]),
              static_cast<unsigned long long>(dims[i]));
        H5Sclose(dataspace);
        H5Dclose(dataset_id);
        return false;
      }
    }

    // Select the hyperslab in the file dataspace
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_array.data(), nullptr,
                        count_array.data(), nullptr);

    // Define the memory dataspace for contiguous reading
    hsize_t total_elements = std::accumulate(
        count_array.begin(), count_array.end(), 1, std::multiplies<hsize_t>());
    hid_t memspace = H5Screate_simple(1, &total_elements, nullptr);

    // Resize data to hold the read elements
    data.resize(total_elements);
    herr_t status = H5Dread(dataset_id, getHDF5Type<T>(), memspace, dataspace,
                            H5P_DEFAULT, data.data());

    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Dclose(dataset_id);

    return status >= 0;
  }

private:
  bool file_open;
  bool file_closed;

  /**
   * @brief Maps C++ data types to HDF5 native data types for the C API
   *
   * @tparam T The C++ data type
   * @return The corresponding HDF5 native data type
   */
  template <typename T> hid_t getHDF5Type();
};

// Template specializations for HDF5 type mappings
template <> hid_t HDF5Helper::getHDF5Type<int64_t>() {
  return H5T_NATIVE_INT64;
}
template <> hid_t HDF5Helper::getHDF5Type<double>() {
  return H5T_NATIVE_DOUBLE;
}
template <> hid_t HDF5Helper::getHDF5Type<int>() { return H5T_NATIVE_INT; }
template <> hid_t HDF5Helper::getHDF5Type<int[3]>() { return H5T_NATIVE_INT; }
template <> hid_t HDF5Helper::getHDF5Type<int[6]>() { return H5T_NATIVE_INT; }
template <> hid_t HDF5Helper::getHDF5Type<double[3]>() {
  return H5T_NATIVE_DOUBLE;
}
template <> hid_t HDF5Helper::getHDF5Type<double *>() {
  return H5T_NATIVE_DOUBLE;
}

void writeGridFileSerial(std::shared_ptr<Cell> *cells) {

  // Get the metadata
  Metadata &metadata = Metadata::getInstance();

  // Get the output filepath
  const std::string filename = metadata.output_file;

  message("Writing grid data to %s", filename.c_str());

  // Create a new HDF5 file
  HDF5Helper hdf5(filename, H5F_ACC_TRUNC);

  // Create the Header group and write out the metadata
  hdf5.createGroup("Header");
  hdf5.writeAttribute<int>("Header", "NGridPoint", metadata.n_grid_points);
  hdf5.writeAttribute<int[3]>("Header", "Simulation_CDim", metadata.cdim);
  hdf5.writeAttribute<double[3]>("Header", "BoxSize", metadata.dim);
  hdf5.writeAttribute<double>("Header", "MaxKernelRadius",
                              metadata.max_kernel_radius);
  hdf5.writeAttribute<double>("Header", "Redshift", metadata.redshift);

  // Create the Grids group
  hdf5.createGroup("Grids");

  // Loop over cells and collect how many grid points we have in each cell
  std::vector<int> grid_point_counts(metadata.nr_cells, 0);
  for (int cid = 0; cid < metadata.nr_cells; cid++) {
    grid_point_counts[cid] = cells[cid]->grid_points.size();
  }

  // Now we have the counts convert these to a start index for each cell so we
  // can use a cell look up table to find the grid points
  std::vector<int> grid_point_start(metadata.nr_cells, 0);
  for (int cid = 1; cid < metadata.nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Write out this cell lookup table
  std::array<hsize_t, 1> sim_cell_dims = {
      static_cast<hsize_t>(metadata.nr_cells)};
  hdf5.createGroup("Cells");
  hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                            sim_cell_dims);
  hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                            sim_cell_dims);

  // Create a dataset we'll write the grid positions into
  std::array<hsize_t, 2> grid_point_positions_dims = {
      static_cast<hsize_t>(metadata.n_grid_points), static_cast<hsize_t>(3)};
  hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                grid_point_positions_dims);

  // We only want to write the positions once so lets make a flag to ensure
  // we only do this once
  bool written_positions = false;

  // Loop over the kernels and write out the grids themselves
  for (double kernel_rad : metadata.kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Create the kernel group
    hdf5.createGroup("Grids/" + kernel_name);

    // Create the grid point over densities dataset
    std::array<hsize_t, 1> grid_point_overdens_dims = {
        static_cast<hsize_t>(metadata.n_grid_points)};
    hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                      "/GridPointOverDensities",
                                  grid_point_overdens_dims);

    // Write out the grid data cell by cell
    for (int cid = 0; cid < metadata.nr_cells; cid++) {
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
        cell_grid_ovdens.push_back(gp->getOverDensity(kernel_rad));

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
    hdf5.writeAttribute<int>("Header", "NGridPoint", metadata.n_grid_points);
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
  std::vector<int> rank_grid_point_counts(metadata.nr_cells, 0);
  for (int cid = 0; cid < metadata.nr_cells; cid++) {
    // Only count cells that are owned by the current rank
    if (cells[cid]->rank == metadata.rank) {
      rank_grid_point_counts[cid] = cells[cid]->grid_points.size();
    }
  }

  // Aggregate the grid point counts across all ranks using MPI_Allreduce
  // Each rank ends up with a complete view of grid_point_counts across all
  // cells
  std::vector<int> grid_point_counts(metadata.nr_cells, 0);
  MPI_Allreduce(rank_grid_point_counts.data(), grid_point_counts.data(),
                metadata.nr_cells, MPI_INT, MPI_SUM, comm);

  // Calculate starting indices for each cell in the global grid point array
  std::vector<int> grid_point_start(metadata.nr_cells, 0);
  for (int cid = 1; cid < metadata.nr_cells; cid++) {
    grid_point_start[cid] =
        grid_point_start[cid - 1] + grid_point_counts[cid - 1];
  }

  // Only rank 0 writes the lookup table to map cells to grid points
  if (metadata.rank == 0) {
    std::array<hsize_t, 1> cell_dims = {
        static_cast<hsize_t>(metadata.nr_cells)};
    hdf5.createGroup("Cells");
    hdf5.writeDataset<int, 1>("Cells/GridPointStart", grid_point_start,
                              cell_dims);
    hdf5.writeDataset<int, 1>("Cells/GridPointCounts", grid_point_counts,
                              cell_dims);

    // Define dimensions for grid point positions dataset
    std::array<hsize_t, 2> grid_point_positions_dims = {
        static_cast<hsize_t>(metadata.n_grid_points), static_cast<hsize_t>(3)};
    hdf5.createDataset<double, 2>("Grids/GridPointPositions",
                                  grid_point_positions_dims);
  }

  // Track if grid point positions have been written to avoid redundancy
  bool written_positions = false;

  // Iterate over kernel radii to write grid data for each kernel
  for (double kernel_rad : metadata.kernel_radii) {
    std::string kernel_name = "Kernel_" + std::to_string(kernel_rad);

    // Rank 0 creates the group for each kernel in the Grids group
    if (metadata.rank == 0) {
      hdf5.createGroup("Grids/" + kernel_name);
    }

    // Rank 0 also defines the dataset for storing grid point over-densities
    if (metadata.rank == 0) {
      std::array<hsize_t, 1> grid_point_overdens_dims = {
          static_cast<hsize_t>(metadata.n_grid_points)};
      hdf5.createDataset<double, 1>("Grids/" + kernel_name +
                                        "/GridPointOverDensities",
                                    grid_point_overdens_dims);
    }

    // Synchronize all ranks to ensure consistent data access
    MPI_Barrier(comm);

// Process cells and write grid data in parallel using OpenMP
#pragma omp parallel for
    for (int cid = 0; cid < metadata.nr_cells; cid++) {
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

#endif // HDF_IO_H_
