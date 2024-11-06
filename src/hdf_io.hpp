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
#endif // HDF_IO_H_
