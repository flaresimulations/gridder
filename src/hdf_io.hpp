/**
 * @file hdf_io.hpp
 * @brief Header file for the HDF5 I/O helper class with proper serial/parallel
 * support
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

// Forward declarations
class Simulation;
class Grid;

/**
 * @brief Base HDF5 helper class for serial I/O operations
 *
 * This class handles basic HDF5 operations without MPI dependencies.
 */
class HDF5Helper {
public:
  hid_t file_id; ///< HDF5 file identifier

  /**
   * @brief Constructor for HDF5 helper object
   *
   * @param filename The name of the HDF5 file to open/create
   * @param accessMode The file access mode (H5F_ACC_RDONLY, H5F_ACC_RDWR,
   * H5F_ACC_TRUNC, etc.)
   */
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY);

  /**
   * @brief Destructor - ensures proper cleanup
   */
  ~HDF5Helper();

  /**
   * @brief Manually close the HDF5 file
   */
  void close();

  /**
   * @brief Check if file is currently open
   * @return true if file is open, false otherwise
   */
  bool isOpen() const { return file_open; }

  /**
   * @brief Create a group in the HDF5 file
   * @param groupName Name of the group to create
   * @return true if successful, false otherwise
   */
  bool createGroup(const std::string &groupName);

  /**
   * @brief Open an existing group in the HDF5 file
   * @param groupName Name of the group to open
   * @return Group identifier if successful, negative value on error
   */
  hid_t openGroup(const std::string &groupName);

  /**
   * @brief Close a group
   * @param group_id Group identifier to close
   */
  void closeGroup(hid_t group_id);

  /**
   * @brief Read an attribute from an HDF5 object
   * @tparam T Data type of the attribute
   * @param objName Name of the object containing the attribute
   * @param attributeName Name of the attribute
   * @param attributeValue Reference to store the attribute value
   * @return true if successful, false otherwise
   */
  template <typename T>
  bool readAttribute(const std::string &objName,
                     const std::string &attributeName, T &attributeValue);

  /**
   * @brief Write an attribute to an HDF5 object
   * @tparam T Data type of the attribute
   * @param objName Name of the object to attach the attribute to
   * @param attributeName Name of the attribute
   * @param attributeValue Value to write
   * @return true if successful, false otherwise
   */
  template <typename T>
  bool writeAttribute(const std::string &objName,
                      const std::string &attributeName,
                      const T &attributeValue);

  /**
   * @brief Read a complete dataset
   * @tparam T Data type of dataset elements
   * @param datasetName Name of the dataset
   * @param data Vector to store the data
   * @return true if successful, false otherwise
   */
  template <typename T>
  bool readDataset(const std::string &datasetName, std::vector<T> &data);

  /**
   * @brief Create a dataset with specified dimensions
   * @tparam T Data type of dataset elements
   * @tparam Rank Number of dimensions
   * @param datasetName Name of the dataset
   * @param dims Dimensions of the dataset
   * @return true if successful, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool createDataset(const std::string &datasetName,
                     const std::array<hsize_t, Rank> &dims);

  /**
   * @brief Write a complete dataset
   * @tparam T Data type of dataset elements
   * @tparam Rank Number of dimensions
   * @param datasetName Name of the dataset
   * @param data Data to write
   * @param dims Dimensions of the dataset
   * @return true if successful, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool writeDataset(const std::string &datasetName, const std::vector<T> &data,
                    const std::array<hsize_t, Rank> &dims);

  /**
   * @brief Write a slice of data to an existing dataset
   * @tparam T Data type of dataset elements
   * @tparam Rank Number of dimensions
   * @param datasetName Name of the dataset
   * @param data Data to write
   * @param start Starting indices for the hyperslab
   * @param count Size of the hyperslab in each dimension
   * @return true if successful, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool writeDatasetSlice(const std::string &datasetName,
                         const std::vector<T> &data,
                         const std::array<hsize_t, Rank> &start,
                         const std::array<hsize_t, Rank> &count);

  /**
   * @brief Read a slice of data from a dataset
   * @tparam T Data type of dataset elements
   * @tparam Rank Number of dimensions
   * @param datasetName Name of the dataset
   * @param data Vector to store the data
   * @param start Starting indices for the hyperslab
   * @param count Size of the hyperslab in each dimension
   * @return true if successful, false otherwise
   */
  template <typename T, std::size_t Rank>
  bool readDatasetSlice(const std::string &datasetName, std::vector<T> &data,
                        const std::array<hsize_t, Rank> &start,
                        const std::array<hsize_t, Rank> &count);

  /**
   * @brief Check if a dataset is virtual
   * @param datasetName Name of the dataset
   * @return true if dataset is virtual, false otherwise
   */
  bool isVirtualDataset(const std::string &datasetName);

#ifdef WITH_MPI
  /**
   * @brief Get the MPI rank (only available in MPI builds)
   * @return MPI rank
   */
  int getMPIRank() const { return mpi_rank; }

  /**
   * @brief Get the MPI size (only available in MPI builds)
   * @return MPI size
   */
  int getMPISize() const { return mpi_size; }
#endif

protected:
  bool file_open; ///< Track if file is open

#ifdef WITH_MPI
  int mpi_rank; ///< MPI rank
  int mpi_size; ///< MPI size
#endif

  /**
   * @brief Map C++ types to HDF5 native types
   * @tparam T C++ data type
   * @return Corresponding HDF5 native type
   */
  template <typename T> hid_t getHDF5Type();

  /**
   * @brief Create transfer property list for I/O operations (always serial)
   * @return Property list identifier
   */
  hid_t createTransferPlist();

  /**
   * @brief Validate that a slice is within dataset bounds
   * @tparam Rank Number of dimensions
   * @param dataset_id Dataset identifier
   * @param start Starting indices
   * @param count Size in each dimension
   * @return true if valid, false otherwise
   */
  template <std::size_t Rank>
  bool validateSliceBounds(hid_t dataset_id,
                           const std::array<hsize_t, Rank> &start,
                           const std::array<hsize_t, Rank> &count);
};

// Function prototypes for grid output (implemented in output.cpp)
void writeGridFileSerial(Simulation *sim, Grid *grid);
#ifdef WITH_MPI
void writeGridFileParallel(Simulation *sim, Grid *grid);
void createVirtualFile(const std::string &base_filename, int num_ranks,
                       Simulation *sim, Grid *grid);
#endif

// Template implementations

template <typename T>
bool HDF5Helper::readAttribute(const std::string &objName,
                               const std::string &attributeName,
                               T &attributeValue) {
  if (!file_open) {
    error("Cannot read attribute: file is not open");
    return false;
  }

  hid_t obj_id = H5Oopen(file_id, objName.c_str(), H5P_DEFAULT);
  if (obj_id < 0) {
    error("Failed to open object '%s'", objName.c_str());
    return false;
  }

  hid_t attr_id = H5Aopen(obj_id, attributeName.c_str(), H5P_DEFAULT);
  if (attr_id < 0) {
    error("Failed to open attribute '%s' in object '%s'", attributeName.c_str(),
          objName.c_str());
    H5Oclose(obj_id);
    return false;
  }

  hid_t attr_type = getHDF5Type<T>();
  herr_t status = H5Aread(attr_id, attr_type, &attributeValue);

  H5Aclose(attr_id);
  H5Oclose(obj_id);

  if (status < 0) {
    error("Failed to read attribute '%s' from object '%s'",
          attributeName.c_str(), objName.c_str());
    return false;
  }

  return true;
}

template <typename T>
bool HDF5Helper::writeAttribute(const std::string &objName,
                                const std::string &attributeName,
                                const T &attributeValue) {
  if (!file_open) {
    error("Cannot write attribute: file is not open");
    return false;
  }

  hid_t obj_id = H5Oopen(file_id, objName.c_str(), H5P_DEFAULT);
  if (obj_id < 0) {
    error("Failed to open object '%s'", objName.c_str());
    return false;
  }

  hid_t attr_type = getHDF5Type<T>();
  hid_t dataspace_id;

  // Handle arrays vs scalars
  if constexpr (std::is_array_v<T>) {
    hsize_t dims[1] = {std::extent_v<T>};
    dataspace_id = H5Screate_simple(1, dims, nullptr);
  } else {
    dataspace_id = H5Screate(H5S_SCALAR);
  }

  if (dataspace_id < 0) {
    error("Failed to create dataspace for attribute '%s'",
          attributeName.c_str());
    H5Oclose(obj_id);
    return false;
  }

  hid_t attr_id = H5Acreate(obj_id, attributeName.c_str(), attr_type,
                            dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  if (attr_id < 0) {
    error("Failed to create attribute '%s' in object '%s'",
          attributeName.c_str(), objName.c_str());
    H5Sclose(dataspace_id);
    H5Oclose(obj_id);
    return false;
  }

  herr_t status = H5Awrite(attr_id, attr_type, &attributeValue);

  H5Aclose(attr_id);
  H5Sclose(dataspace_id);
  H5Oclose(obj_id);

  if (status < 0) {
    error("Failed to write attribute '%s' to object '%s'",
          attributeName.c_str(), objName.c_str());
    return false;
  }

  return true;
}

template <typename T>
bool HDF5Helper::readDataset(const std::string &datasetName,
                             std::vector<T> &data) {
  if (!file_open) {
    error("Cannot read dataset: file is not open");
    return false;
  }

  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    error("Failed to open dataset '%s'", datasetName.c_str());
    return false;
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  int rank = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(rank);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  hsize_t total_elements =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<hsize_t>());
  data.resize(total_elements);

  hid_t plist_id = createTransferPlist();
  herr_t status = H5Dread(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                          plist_id, data.data());

  H5Pclose(plist_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);

  if (status < 0) {
    error("Failed to read dataset '%s'", datasetName.c_str());
    return false;
  }

  return true;
}

template <typename T, std::size_t Rank>
bool HDF5Helper::createDataset(const std::string &datasetName,
                               const std::array<hsize_t, Rank> &dims) {
  if (!file_open) {
    error("Cannot create dataset: file is not open");
    return false;
  }

  hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), nullptr);
  if (dataspace_id < 0) {
    error("Failed to create dataspace for dataset '%s'", datasetName.c_str());
    return false;
  }

  hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

  // Set chunking for better performance
  std::array<hsize_t, Rank> chunk_dims = dims;
  for (auto &chunk_dim : chunk_dims) {
    chunk_dim = std::min(chunk_dim,
                         static_cast<hsize_t>(1024)); // Reasonable chunk size
  }
  H5Pset_chunk(plist_id, Rank, chunk_dims.data());

  hid_t dataset_id =
      H5Dcreate(file_id, datasetName.c_str(), getHDF5Type<T>(), dataspace_id,
                H5P_DEFAULT, plist_id, H5P_DEFAULT);

  H5Pclose(plist_id);
  H5Sclose(dataspace_id);

  if (dataset_id < 0) {
    error("Failed to create dataset '%s'", datasetName.c_str());
    return false;
  }

  H5Dclose(dataset_id);
  return true;
}

template <typename T, std::size_t Rank>
bool HDF5Helper::writeDataset(const std::string &datasetName,
                              const std::vector<T> &data,
                              const std::array<hsize_t, Rank> &dims) {
  if (!file_open) {
    error("Cannot write dataset: file is not open");
    return false;
  }

  // Verify data size matches dimensions
  hsize_t expected_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<hsize_t>());
  if (data.size() != expected_size) {
    error("Data size (%zu) doesn't match expected size (%llu) for dataset '%s'",
          data.size(), static_cast<unsigned long long>(expected_size),
          datasetName.c_str());
    return false;
  }

  hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), nullptr);
  if (dataspace_id < 0) {
    error("Failed to create dataspace for dataset '%s'", datasetName.c_str());
    return false;
  }

  hid_t dataset_id =
      H5Dcreate(file_id, datasetName.c_str(), getHDF5Type<T>(), dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_id < 0) {
    error("Failed to create dataset '%s'", datasetName.c_str());
    H5Sclose(dataspace_id);
    return false;
  }

  hid_t plist_id = createTransferPlist();
  herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                           plist_id, data.data());

  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  if (status < 0) {
    error("Failed to write dataset '%s'", datasetName.c_str());
    return false;
  }

  return true;
}

template <typename T, std::size_t Rank>
bool HDF5Helper::writeDatasetSlice(const std::string &datasetName,
                                   const std::vector<T> &data,
                                   const std::array<hsize_t, Rank> &start,
                                   const std::array<hsize_t, Rank> &count) {
  if (!file_open) {
    error("Cannot write dataset slice: file is not open");
    return false;
  }

  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    error("Failed to open dataset '%s'", datasetName.c_str());
    return false;
  }

  // Validate slice bounds
  if (!validateSliceBounds(dataset_id, start, count)) {
    H5Dclose(dataset_id);
    return false;
  }

  // Verify data size matches slice size
  hsize_t expected_size = std::accumulate(count.begin(), count.end(), 1,
                                          std::multiplies<hsize_t>());
  if (data.size() != expected_size) {
    error("Data size (%zu) doesn't match slice size (%llu) for dataset '%s'",
          data.size(), static_cast<unsigned long long>(expected_size),
          datasetName.c_str());
    H5Dclose(dataset_id);
    return false;
  }

  hid_t filespace_id = H5Dget_space(dataset_id);
  herr_t select_status =
      H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start.data(), nullptr,
                          count.data(), nullptr);
  if (select_status < 0) {
    error("Failed to select hyperslab for dataset '%s'", datasetName.c_str());
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return false;
  }

  hid_t memspace_id = H5Screate_simple(Rank, count.data(), nullptr);
  if (memspace_id < 0) {
    error("Failed to create memory space for dataset slice '%s'",
          datasetName.c_str());
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return false;
  }

  hid_t plist_id = createTransferPlist();
  herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), memspace_id,
                           filespace_id, plist_id, data.data());

  H5Pclose(plist_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);
  H5Dclose(dataset_id);

  if (status < 0) {
    error("Failed to write slice to dataset '%s'", datasetName.c_str());
    return false;
  }

  return true;
}

template <typename T, std::size_t Rank>
bool HDF5Helper::readDatasetSlice(const std::string &datasetName,
                                  std::vector<T> &data,
                                  const std::array<hsize_t, Rank> &start,
                                  const std::array<hsize_t, Rank> &count) {
  if (!file_open) {
    error("Cannot read dataset slice: file is not open");
    return false;
  }

  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    error("Failed to open dataset '%s'", datasetName.c_str());
    return false;
  }

  // Validate slice bounds
  if (!validateSliceBounds(dataset_id, start, count)) {
    H5Dclose(dataset_id);
    return false;
  }

  hid_t filespace_id = H5Dget_space(dataset_id);
  herr_t select_status =
      H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start.data(), nullptr,
                          count.data(), nullptr);
  if (select_status < 0) {
    error("Failed to select hyperslab for dataset '%s'", datasetName.c_str());
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return false;
  }

  hsize_t total_elements = std::accumulate(count.begin(), count.end(), 1,
                                           std::multiplies<hsize_t>());
  
  // Debug: Print resize information
  message("HDF5Helper::readDatasetSlice: Preparing vector for dataset '%s' with %llu elements (%.2f GB)", 
          datasetName.c_str(), total_elements, 
          (total_elements * sizeof(T)) / (1024.0 * 1024.0 * 1024.0));
  
  try {
    data.reserve(total_elements);
    data.resize(total_elements);
  } catch (const std::bad_alloc& e) {
    error("Failed to allocate vector for dataset '%s' with %llu elements (%.2f GB). "
          "Memory allocation failed: %s", 
          datasetName.c_str(), total_elements,
          (total_elements * sizeof(T)) / (1024.0 * 1024.0 * 1024.0), e.what());
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return false;
  }

  hid_t memspace_id = H5Screate_simple(Rank, count.data(), nullptr);
  if (memspace_id < 0) {
    error("Failed to create memory space for dataset slice '%s'",
          datasetName.c_str());
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return false;
  }

  hid_t plist_id = createTransferPlist();
  herr_t status = H5Dread(dataset_id, getHDF5Type<T>(), memspace_id,
                          filespace_id, plist_id, data.data());

  H5Pclose(plist_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);
  H5Dclose(dataset_id);

  if (status < 0) {
    error("Failed to read slice from dataset '%s'", datasetName.c_str());
    return false;
  }

  return true;
}

template <std::size_t Rank>
bool HDF5Helper::validateSliceBounds(hid_t dataset_id,
                                     const std::array<hsize_t, Rank> &start,
                                     const std::array<hsize_t, Rank> &count) {
  hid_t dataspace_id = H5Dget_space(dataset_id);
  int dataset_rank = H5Sget_simple_extent_ndims(dataspace_id);

  if (dataset_rank != static_cast<int>(Rank)) {
    error("Dataset rank (%d) doesn't match template rank (%zu)", dataset_rank,
          Rank);
    H5Sclose(dataspace_id);
    return false;
  }

  std::array<hsize_t, Rank> dims;
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
  H5Sclose(dataspace_id);

  for (std::size_t i = 0; i < Rank; ++i) {
    if (start[i] + count[i] > dims[i]) {
      error("Slice out of bounds in dimension %zu: start=%llu, count=%llu, "
            "dim=%llu",
            i, static_cast<unsigned long long>(start[i]),
            static_cast<unsigned long long>(count[i]),
            static_cast<unsigned long long>(dims[i]));
      return false;
    }
  }

  return true;
}

#endif // HDF_IO_H_
