/*******************************************************************************
 * HDF5Helper - A helper class for parallel HDF5 file operations using the C API
 *
 * This file is part of a parallel I/O utility for handling HDF5 files in an
 * MPI environment. Due to limitations in many HPC systems, parallel HDF5
 * installations often lack support for the C++ API, necessitating the use of
 * the C API.
 *
 * The HDF5 C API provides robust support for parallel I/O with MPI and is
 * well-suited for high-performance applications where datasets are accessed
 * across distributed processes.
 *
 * This class is designed to simplify the usage of the HDF5 C API by providing
 * high-level methods for file access, dataset creation, attribute handling,
 * and data reading/writing.
 ******************************************************************************/

#ifndef PARALLEL_IO_H_
#define PARALLEL_IO_H_

#include <array>
#include <hdf5.h> // HDF5 C API
#include <mpi.h>
#include <string>
#include <type_traits>
#include <vector>

// Local includes
#include "logger.hpp"

class HDF5Helper {
public:
  hid_t file_id; ///< HDF5 file identifier
  MPI_Comm comm; ///< MPI communicator
  MPI_Info info; ///< MPI info object for additional options

  /**
   * @brief Constructor for parallel HDF5 with MPI support
   *
   * This constructor initializes the HDF5 file in parallel mode using
   * MPI for distributed I/O. It creates a file access property list
   * configured for MPI I/O and opens the specified file.
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
    H5Pclose(fapl); // Close the property list after use
  }

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

  // Additional methods (e.g., readDataset, readAttribute) should be defined
  // similarly to ensure complete functionality with the C API.

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
template <> hid_t HDF5Helper::getHDF5Type<double[3]>() {
  return H5T_NATIVE_DOUBLE;
}
template <> hid_t HDF5Helper::getHDF5Type<double *>() {
  return H5T_NATIVE_DOUBLE;
}

#endif // PARALLEL_IO_H_
