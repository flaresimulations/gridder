/**
 * @file hdf_io.cpp
 * @brief Implementation of the HDF5 I/O helper class with proper
 * serial/parallel support
 */

// Standard includes
#include <hdf5.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "hdf_io.hpp"
#include "logger.hpp"

/**
 * @brief Constructor for HDF5Helper - always uses serial HDF5 operations
 *
 * @param filename The name of the HDF5 file to open/create
 * @param accessMode The file access mode
 */
HDF5Helper::HDF5Helper(const std::string &filename, unsigned int accessMode)
    : file_open(false) {

#ifdef WITH_MPI
  // Get MPI info for rank identification only
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Always use serial HDF5 setup
  if (accessMode == H5F_ACC_RDONLY) {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } else if (accessMode == H5F_ACC_RDWR) {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  } else {
    file_id = H5Fcreate(filename.c_str(), accessMode, H5P_DEFAULT, H5P_DEFAULT);
  }

  if (file_id < 0) {
#ifdef WITH_MPI
    error("Rank %d: Failed to open/create HDF5 file: %s", mpi_rank,
          filename.c_str());
#else
    error("Failed to open/create HDF5 file: %s", filename.c_str());
#endif
    return;
  }

  file_open = true;
}

/**
 * @brief Destructor - ensures proper cleanup
 */
HDF5Helper::~HDF5Helper() {
  if (file_open) {
    close();
  }
}

/**
 * @brief Manually close the HDF5 file
 */
void HDF5Helper::close() {
  if (file_open && file_id >= 0) {
    H5Fclose(file_id);
    file_open = false;
  }
}

/**
 * @brief Create a group in the HDF5 file
 *
 * @param groupName Name of the group to create
 * @return true if successful, false otherwise
 */
bool HDF5Helper::createGroup(const std::string &groupName) {
  if (!file_open) {
    error("Cannot create group: file is not open");
    return false;
  }

  hid_t group_id = H5Gcreate(file_id, groupName.c_str(), H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
  if (group_id < 0) {
    error("Failed to create group '%s'", groupName.c_str());
    return false;
  }
  H5Gclose(group_id);
  return true;
}

/**
 * @brief Open an existing group in the HDF5 file
 *
 * @param groupName Name of the group to open
 * @return Group identifier if successful, negative value on error
 */
hid_t HDF5Helper::openGroup(const std::string &groupName) {
  if (!file_open) {
    error("Cannot open group: file is not open");
    return -1;
  }

  hid_t group_id = H5Gopen(file_id, groupName.c_str(), H5P_DEFAULT);
  if (group_id < 0) {
    error("Failed to open group '%s'", groupName.c_str());
    return -1;
  }

  return group_id; // Return the open group ID - caller must close it
}

/**
 * @brief Close a group
 *
 * @param group_id Group identifier to close
 */
void HDF5Helper::closeGroup(hid_t group_id) {
  if (group_id >= 0) {
    H5Gclose(group_id);
  }
}

/**
 * @brief Check if a dataset is virtual
 *
 * @param datasetName Name of the dataset to check
 * @return true if dataset is virtual, false otherwise
 */
bool HDF5Helper::isVirtualDataset(const std::string &datasetName) {
  if (!file_open) {
    error("Cannot check dataset: file is not open");
    return false;
  }

  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    return false; // Dataset doesn't exist or can't be opened
  }

  hid_t dcpl_id = H5Dget_create_plist(dataset_id);
  bool is_virtual = (H5Pget_layout(dcpl_id) == H5D_VIRTUAL);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);

  return is_virtual;
}

/**
 * @brief Create transfer property list for I/O operations
 *
 * @return Property list identifier
 */
hid_t HDF5Helper::createTransferPlist() {
  // Always return default transfer property list (serial operations)
  return H5Pcreate(H5P_DATASET_XFER);
}

// Template specializations for HDF5 type mappings

/**
 * @brief Template specialization for int64_t
 */
template <> hid_t HDF5Helper::getHDF5Type<int64_t>() {
  return H5T_NATIVE_INT64;
}

/**
 * @brief Template specialization for double
 */
template <> hid_t HDF5Helper::getHDF5Type<double>() {
  return H5T_NATIVE_DOUBLE;
}

/**
 * @brief Template specialization for int
 */
template <> hid_t HDF5Helper::getHDF5Type<int>() { return H5T_NATIVE_INT; }

/**
 * @brief Template specialization for float
 */
template <> hid_t HDF5Helper::getHDF5Type<float>() { return H5T_NATIVE_FLOAT; }

/**
 * @brief Template specialization for int arrays (used for attributes)
 */
template <> hid_t HDF5Helper::getHDF5Type<int[3]>() { return H5T_NATIVE_INT; }

/**
 * @brief Template specialization for int arrays (used for attributes)
 */
template <> hid_t HDF5Helper::getHDF5Type<int[6]>() { return H5T_NATIVE_INT; }

/**
 * @brief Template specialization for double arrays (used for attributes)
 */
template <> hid_t HDF5Helper::getHDF5Type<double[3]>() {
  return H5T_NATIVE_DOUBLE;
}

/**
 * @brief Template specialization for double arrays (used for attributes)
 */
template <> hid_t HDF5Helper::getHDF5Type<double[6]>() {
  return H5T_NATIVE_DOUBLE;
}

/**
 * @brief Template specialization for double pointer (legacy support)
 */
template <> hid_t HDF5Helper::getHDF5Type<double *>() {
  return H5T_NATIVE_DOUBLE;
}

// Conditional specializations to avoid conflicts between different integer
// types On some systems, long == int64_t, on others they're different

#if !defined(__SIZEOF_LONG__) || (__SIZEOF_LONG__ != 8)
/**
 * @brief Template specialization for long (when long != int64_t)
 */
template <> hid_t HDF5Helper::getHDF5Type<long>() { return H5T_NATIVE_LONG; }

/**
 * @brief Template specialization for long arrays (when long != int64_t)
 */
template <> hid_t HDF5Helper::getHDF5Type<long[6]>() { return H5T_NATIVE_LONG; }
#endif

#if !defined(__SIZEOF_LONG__) || (__SIZEOF_LONG__ != __SIZEOF_SIZE_T__)
/**
 * @brief Template specialization for unsigned long (when unsigned long !=
 * size_t)
 */
template <> hid_t HDF5Helper::getHDF5Type<unsigned long>() {
  return H5T_NATIVE_ULONG;
}

/**
 * @brief Template specialization for unsigned long arrays (when unsigned long
 * != size_t)
 */
template <> hid_t HDF5Helper::getHDF5Type<unsigned long[6]>() {
  return H5T_NATIVE_ULONG;
}
#endif

/**
 * @brief Template specialization for size_t
 */
template <> hid_t HDF5Helper::getHDF5Type<size_t>() { return H5T_NATIVE_HSIZE; }

/**
 * @brief Template specialization for size_t arrays (used for attributes)
 */
template <> hid_t HDF5Helper::getHDF5Type<size_t[6]>() {
  return H5T_NATIVE_HSIZE;
}
