// Standard includes
#include <hdf5.h> // HDF5 C API

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "hdf_io.hpp"
#include "logger.hpp"

/**
 * @brief Constructor for a HDF5 helper object with serial I/O
 *
 * @param filename The name of the HDF5 file to open.
 * @param accessMode The file access mode, e.g., H5F_ACC_RDONLY, H5F_ACC_RDWR.
 */
HDF5Helper::HDF5Helper(const std::string &filename, unsigned int accessMode)
    : file_open(true), file_closed(false) {
  if (accessMode == H5F_ACC_RDONLY) {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } else {
    file_id = H5Fcreate(filename.c_str(), accessMode, H5P_DEFAULT, H5P_DEFAULT);
  }

  if (file_id < 0)
    error("Failed to open HDF5 file: ", filename.c_str());
}

/**
 * @brief Destructor closes the file if it is still open
 */
HDF5Helper::~HDF5Helper() {
  if (!file_closed)
    H5Fclose(file_id);
}

/**
 * @brief Closes the HDF5 file, marking it as closed
 */
void HDF5Helper::close() {
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
bool HDF5Helper::createGroup(const std::string &groupName) {
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
hid_t HDF5Helper::openGroup(const std::string &groupName) {
  hid_t group_id = H5Gopen(file_id, groupName.c_str(), H5P_DEFAULT);
  if (group_id < 0)
    return -1;

  H5Gclose(group_id);
  return group_id;
}

/**
 * @brief Checks if the given dataset is a virtual dataset (VDS)
 *
 * @param datasetName The name of the dataset to check
 * @return true if the dataset is virtual, false otherwise
 */
bool HDF5Helper::isVirtualDataset(const std::string &datasetName) {
  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
    return false;

  hid_t dapl_id = H5Dget_create_plist(dataset_id);
  bool is_virtual = (H5Pget_layout(dapl_id) == H5D_VIRTUAL);

  H5Pclose(dapl_id);
  H5Dclose(dataset_id);
  return is_virtual;
}

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
