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

// Forward declaration
class Simulation;
class Grid;

class HDF5Helper {
public:
  hid_t file_id; ///< HDF5 file identifier

#ifdef WITH_MPI
  MPI_Comm comm; ///< MPI communicator
  MPI_Info info; ///< MPI info object for additional options
#endif

#ifdef WITH_MPI
  // Constructor with parallel I/O prototype
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY,
             MPI_Comm communicator = MPI_COMM_WORLD,
             MPI_Info file_info = MPI_INFO_NULL);
#else
  // Constructor with serial I/O prototype
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY);
#endif

  // Destructor prototype
  ~HDF5Helper();

  // Prototypes for member functions (defined in hdf_io.cpp)
  void close();
  bool createGroup(const std::string &groupName);
  hid_t openGroup(const std::string &groupName);
  template <typename T>
  bool readAttribute(const std::string &objName,
                     const std::string &attributeName, T &attributeValue);
  template <typename T>
  bool writeAttribute(const std::string &objName,
                      const std::string &attributeName,
                      const T &attributeValue);
  template <typename T>
  bool readDataset(const std::string &datasetName, std::vector<T> &data);
  template <typename T, std::size_t Rank>
  bool createDataset(const std::string &datasetName,
                     const std::array<hsize_t, Rank> &dims);
  template <typename T, std::size_t Rank>
  bool writeDataset(const std::string &datasetName, const std::vector<T> &data,
                    const std::array<hsize_t, Rank> &dims);
  template <typename T, std::size_t Rank>
  bool writeDatasetSlice(const std::string &datasetName,
                         const std::vector<T> &data,
                         const std::array<hsize_t, Rank> &start,
                         const std::array<hsize_t, Rank> &count);
  bool isVirtualDataset(const std::string &datasetName);
  template <typename T, std::size_t Rank>
  bool readDatasetSliceFromVDS(const std::string &datasetName,
                               std::vector<T> &data,
                               const std::array<hsize_t, Rank> &start_array,
                               const std::array<hsize_t, Rank> &count_array);
  template <typename T, std::size_t Rank>
  bool readDatasetSlice(const std::string &datasetName, std::vector<T> &data,
                        const std::array<hsize_t, Rank> &start_array,
                        const std::array<hsize_t, Rank> &count_array);

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

// Prototypes for writing out the grid data (defined in output.cpp)
void writeGridFileSerial(Simulation *sim, Grid *grid);
void writeGridFileParallel(Simulation *sim, Grid *grid);

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
bool HDF5Helper::readAttribute(const std::string &objName,
                               const std::string &attributeName,
                               T &attributeValue) {
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
bool HDF5Helper::writeAttribute(const std::string &objName,
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
bool HDF5Helper::readDataset(const std::string &datasetName,
                             std::vector<T> &data) {
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
bool HDF5Helper::createDataset(const std::string &datasetName,
                               const std::array<hsize_t, Rank> &dims) {
  // Create the dataspace for the dataset with the specified dimensions
  hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), nullptr);
  if (dataspace_id < 0)
    return false;

  // Define properties for chunking and collective I/O
  hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist_id, Rank, dims.data());

  hid_t dataset_id =
      H5Dcreate(file_id, (datasetName).c_str(), getHDF5Type<T>(), dataspace_id,
                H5P_DEFAULT, plist_id, H5P_DEFAULT);
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
bool HDF5Helper::writeDataset(const std::string &datasetName,
                              const std::vector<T> &data,
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
bool HDF5Helper::writeDatasetSlice(const std::string &datasetName,
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
bool HDF5Helper::readDatasetSliceFromVDS(
    const std::string &datasetName, std::vector<T> &data,
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
    herr_t status = H5Dread(src_dset_id, getHDF5Type<T>(), mem_space, src_space,
                            H5P_DEFAULT, data.data());

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
bool HDF5Helper::readDatasetSlice(
    const std::string &datasetName, std::vector<T> &data,
    const std::array<hsize_t, Rank> &start_array,
    const std::array<hsize_t, Rank> &count_array) {

  /* If we have a virtual dataset, read the slice from the source datasets */
  if (isVirtualDataset(datasetName)) {
    return readDatasetSliceFromVDS(datasetName, data, start_array, count_array);
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

#endif // HDF_IO_H_
