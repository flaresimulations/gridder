/*******************************************************************************
 * HDF5Helper - A helper class for parallel HDF5 file operations
 *
 * This file is part of a parallel I/O utility for handling HDF5 files in
 * an MPI environment. Due to limitations in many HPC systems, parallel
 * HDF5 installations frequently lack support for the C++ API, requiring the
 * use of the C API instead.
 *
 * The HDF5 C API provides robust support for parallel I/O with MPI and offers
 * full functionality required for handling datasets, groups, and attributes,
 * making it suitable for high-performance applications.
 *
 * This class is designed to simplify the usage of the HDF5 C API by providing
 * high-level methods for file access, dataset creation, and data
 *reading/writing.
 ******************************************************************************/

#ifndef PARALLEL_IO_H_
#define PARALLEL_IO_H_

#include <array>
#include <hdf5.h> // C API for HDF5
#include <mpi.h>
#include <string>
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
    // Create the dataspace with the specified dimensions
    hid_t dataspace_id = H5Screate_simple(Rank, dims.data(), NULL);
    if (dataspace_id < 0)
      return false;

    // Create the dataset with the default property list
    hid_t dataset_id =
        H5Dcreate(file_id, datasetName.c_str(), getHDF5Type<T>(), dataspace_id,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
      H5Sclose(dataspace_id);
      return false;
    }

    // Write the data to the dataset
    herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    return status >= 0;
  }

  /**
   * @brief Writes a slice of data to an existing dataset
   *
   * This function writes a slice of data to a specific hyperslab within an
   * existing dataset, allowing parallel processes to write to separate
   * portions.
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
    // Open the dataset from the file
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    // Get the dataspace for the dataset and select a hyperslab
    hid_t filespace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start.data(), NULL,
                        count.data(), NULL);

    // Define the memory dataspace based on count
    hid_t memspace_id = H5Screate_simple(Rank, count.data(), NULL);

    // Set the transfer property list for collective I/O
    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    // Write the data to the selected hyperslab
    herr_t status = H5Dwrite(dataset_id, getHDF5Type<T>(), memspace_id,
                             filespace_id, xfer_plist_id, data.data());

    H5Pclose(xfer_plist_id);
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    return status >= 0;
  }

  /**
   * @brief Reads a complete dataset from the file
   *
   * This function reads data from a dataset with the specified name into
   * a vector.
   *
   * @tparam T Data type of the dataset
   * @param datasetName Name of the dataset to read from
   * @param data The vector to store the read data
   * @return true if the dataset was read successfully, false otherwise
   */
  template <typename T>
  bool readDataset(const std::string &datasetName, std::vector<T> &data) {
    // Open the dataset from the file
    hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
      return false;

    // Get the dataspace and dimensions of the dataset
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    data.resize(dims[0]);

    // Read the data from the dataset
    herr_t status = H5Dread(dataset_id, getHDF5Type<T>(), H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, data.data());
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return status >= 0;
  }

  /**
   * @brief Reads an attribute from an HDF5 object
   *
   * @tparam T Data type of the attribute
   * @param objName The name of the HDF5 object containing the attribute
   * @param attributeName The name of the attribute to read
   * @param attributeValue Variable to store the read attribute value
   * @return true if the attribute was read successfully, false otherwise
   */
  template <typename T>
  bool readAttribute(const std::string &objName,
                     const std::string &attributeName, T &attributeValue) {
    // Open the HDF5 object
    hid_t obj_id = H5Oopen(file_id, objName.c_str(), H5P_DEFAULT);
    if (obj_id < 0)
      return false;

    // Open the attribute and read its value
    hid_t attr_id = H5Aopen(obj_id, attributeName.c_str(), H5P_DEFAULT);
    if (attr_id < 0) {
      H5Oclose(obj_id);
      return false;
    }

    herr_t status = H5Aread(attr_id, getHDF5Type<T>(), &attributeValue);
    H5Aclose(attr_id);
    H5Oclose(obj_id);
    return status >= 0;
  }

private:
  bool file_open;   ///< Flag indicating if the file is currently open
  bool file_closed; ///< Flag indicating if the file has been closed

  /**
   * @brief Maps C++ data types to HDF5 native data types
   *
   * This helper function is specialized for each supported data type.
   *
   * @tparam T The C++ data type
   * @return The corresponding HDF5 native data type
   */
  template <typename T> hid_t getHDF5Type();
};

// Template specializations for supported types
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
