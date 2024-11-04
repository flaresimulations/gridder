#ifndef PARALLEL_IO_H_
#define PARALLEL_IO_H_

#include <H5Cpp.h>
#include <H5public.h>
#include <mpi.h>
#include <string>
#include <vector>

// Local includes
#include "logger.hpp"

class HDF5Helper {
public:
  H5::H5File file;
  MPI_Comm comm;
  MPI_Info info;

  // Constructor for parallel HDF5 with MPI
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY,
             MPI_Comm communicator = MPI_COMM_WORLD,
             MPI_Info file_info = MPI_INFO_NULL)
      : comm(communicator), info(file_info), file_open(true),
        file_closed(false) {
    // Set up the HDF5 file access property list for parallel I/O
    H5::FileAccPropList fapl;
    fapl.setFaplMpio(communicator, file_info);

    // Open the HDF5 file in parallel mode
    file =
        H5::H5File(filename, accessMode, H5::FileCreatPropList::DEFAULT, fapl);
  }

  ~HDF5Helper() {
    if (!this->file_closed)
      this->file.close();
  }

  void close() {
    this->file_closed = true;
    this->file_open = false;
    this->file.close();
  }

  bool createGroup(const std::string &groupName) {
    try {
      H5::Group group = file.createGroup(groupName);
      group.close();
      return true;
    } catch (H5::Exception &e) {
      return false;
    }
  }

  H5::Group openGroup(const std::string &groupName) {
    try {
      H5::Group group = file.openGroup(groupName);
      group.close();
      return group;
    } catch (H5::Exception &e) {
      return H5::Group();
    }
  }

  template <typename T, std::size_t Rank>
  bool writeDataset(const std::string &datasetName, const std::vector<T> &data,
                    const std::array<hsize_t, Rank> &dims) {
    try {
      H5::DataSpace dataspace(Rank, dims.data());
      H5::DSetCreatPropList plist;
      plist.setChunk(Rank,
                     dims.data()); // Set chunking for efficient parallel I/O

      H5::DataSet dataset =
          file.createDataSet(datasetName, getHDF5Type<T>(), dataspace, plist);
      dataset.write(data.data(), getHDF5Type<T>());
      dataspace.close();
      dataset.close();
      return true;
    } catch (H5::Exception &e) {
      return false;
    }
  }

  template <typename T, std::size_t Rank>
  bool writeDatasetSlice(const std::string &datasetName,
                         const std::vector<T> &data,
                         const std::array<hsize_t, Rank> &start,
                         const std::array<hsize_t, Rank> &count) {
    try {
      // Open the existing dataset from the file.
      H5::DataSet dataset = file.openDataSet(datasetName);

      // Get the dataspace for the dataset and select a hyperslab
      H5::DataSpace filespace = dataset.getSpace();
      filespace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

      // Define the memory dataspace based on count
      H5::DataSpace memspace(Rank, count.data());

      // Set the transfer property list for parallel I/O
      H5::DSetMemXferPropList xfer_plist;
      xfer_plist.setDxplMpio(H5FD_MPIO_COLLECTIVE);

      // Write the data to the selected hyperslab in the dataset
      dataset.write(data.data(), getHDF5Type<T>(), memspace, filespace,
                    xfer_plist);

      return true;
    } catch (const H5::Exception &e) {
      error(e.getCDetailMsg());
      return false;
    }
  }

  // Other methods like readDataset, readDatasetSlice, readAttribute, etc.,
  // remain the same.

private:
  bool file_open;
  bool file_closed;

  template <typename T> H5::PredType getHDF5Type();
};

// Ensure you include the template specializations for getHDF5Type() here
// Template specializations for supported types
template <> H5::PredType HDF5Helper::getHDF5Type<int64_t>() {
  return H5::PredType::NATIVE_INT64;
}

template <> H5::PredType HDF5Helper::getHDF5Type<double>() {
  return H5::PredType::NATIVE_DOUBLE;
}

template <> H5::PredType HDF5Helper::getHDF5Type<int>() {
  return H5::PredType::NATIVE_INT;
}

template <> H5::PredType HDF5Helper::getHDF5Type<int[3]>() {
  return H5::PredType::NATIVE_INT;
}

template <> H5::PredType HDF5Helper::getHDF5Type<double[3]>() {
  return H5::PredType::NATIVE_DOUBLE;
}

template <> H5::PredType HDF5Helper::getHDF5Type<double *>() {
  return H5::PredType::NATIVE_DOUBLE;
}

#endif // PARALLEL_IO_H_
