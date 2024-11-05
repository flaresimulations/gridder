#ifndef SERIAL_IO_H_
#define SERIAL_IO_H_

// Standard includes
#include <H5Cpp.h>
#include <numeric>
#include <string>
#include <vector>

// Local includes
#include "logger.hpp"

class HDF5Helper {
public:
  H5::H5File file;

  /** @brief The constructor for the #HDF5Helper class.
   *
   * This will create an instance of the HDF5Helper class with the HDF5 file
   * opened in the specified mode.
   *
   * Note: HDF5 file is closed when the deconstructor is called.
   *
   * @param filename The filepath to the HDF5 file to read or write.
   * @param accessMode The mode with which to open the HDF5 file. Can be
   *                   H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_TRUNC,
   * H5F_ACC_EXCL, or H5F_ACC_CREAT. Defaults to H5F_ACC_RDONLY.
   */
  HDF5Helper(const std::string &filename,
             unsigned int accessMode = H5F_ACC_RDONLY)
      : file(filename, accessMode), file_open(true), file_closed(false) {}
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

  template <typename T>
  bool writeAttribute(const std::string &objName,
                      const std::string &attributeName, T &attributeValue) {
    try {
      H5::Group group(file.openGroup(objName));

      H5::DataSpace dataspace;
      void *data_ptr;

      if constexpr (std::is_array<T>::value) {
        // T is an array
        constexpr std::size_t array_size = std::extent<T>::value;
        hsize_t dims[1] = {array_size};
        dataspace = H5::DataSpace(1, dims);
        data_ptr = static_cast<void *>(&attributeValue[0]); // Corrected line
      } else {
        // T is scalar
        dataspace = H5::DataSpace(H5S_SCALAR);
        data_ptr = &attributeValue;
      }

      H5::Attribute attr = group.createAttribute(
          attributeName, getHDF5Type<std::remove_extent_t<T>>(), dataspace);
      attr.write(getHDF5Type<std::remove_extent_t<T>>(), data_ptr);

      group.close();
      dataspace.close();
      attr.close();
      return true;
    } catch (H5::Exception &e) {
      // Handle exception if necessary
      return false;
    }
  }

  template <typename T, std::size_t Rank>
  bool createDataset(const std::string &datasetName,
                     const std::array<hsize_t, Rank> &dims) {
    try {
      // Create the data space for the dataset with given dimensions.
      H5::DataSpace dataspace(Rank, dims.data());

      // Create the dataset within the file using the defined data space.
      H5::DataSet dataset =
          file.createDataSet(datasetName, getHDF5Type<T>(), dataspace);

      // Optionally set dataset properties or attributes here.

      return true;
    } catch (const H5::Exception &e) {
      // Error handling: print the error message or log it.
      error(e.getCDetailMsg());
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

      // Get the dataspace for the dataset.
      H5::DataSpace filespace = dataset.getSpace();

      // Select the hyperslab in the file dataspace.
      filespace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

      // Define the memory dataspace based on count.
      H5::DataSpace memspace(Rank, count.data());

      // Write the data to the selected hyperslab in the dataset.
      dataset.write(data.data(), getHDF5Type<T>(), memspace, filespace);

      return true;
    } catch (const H5::Exception &e) {
      error(e.getCDetailMsg());
      return false;
    }
  }

  template <typename T, std::size_t Rank>
  bool writeDataset(const std::string &datasetName, const std::vector<T> &data,
                    const std::array<hsize_t, Rank> &dims) {
    try {
      // Create the dataset with the given dimensions.
      createDataset<T, Rank>(datasetName, dims);

      // Write the data as a "slice" over the whole dataset.
      writeDatasetSlice<T, Rank>(datasetName, data, {0}, dims);

      return true;
    } catch (const H5::Exception &e) {
      error(e.getCDetailMsg());
      return false;
    }
  }

  template <typename T>
  bool readDataset(const std::string &datasetName, std::vector<T> &data) {
    try {
      H5::DataSet dataset = file.openDataSet(datasetName);
      H5::DataSpace dataspace = dataset.getSpace();
      std::vector<T> buffer(dataspace.getSelectNpoints());
      dataset.read(buffer.data(), getHDF5Type<T>());
      data = buffer;
      dataspace.close();
      dataset.close();
      return true;
    } catch (H5::Exception &e) {
      return false;
    }
  }

  template <typename T>
  bool readDatasetSlice(const std::string &datasetName, std::vector<T> &data,
                        const hsize_t start, const hsize_t count) {
    try {
      // Open the dataset and retrieve its dataspace
      H5::DataSet dataset = this->file.openDataSet(datasetName);
      H5::DataSpace dataspace = dataset.getSpace();

      // Get the rank and dimensions of the dataset
      int rank = dataspace.getSimpleExtentNdims();
      std::vector<hsize_t> dims(rank);
      dataspace.getSimpleExtentDims(dims.data(), NULL);

      // Calculate total number of elements in the dataset
      hsize_t total_elements = std::accumulate(dims.begin(), dims.end(), 1,
                                               std::multiplies<hsize_t>());

      // Ensure the requested range is within bounds
      if (start + count > total_elements) {
        error("Requested slice (start=%llu, count=%llu) is out of dataset "
              "bounds (total_elements=%llu).",
              static_cast<unsigned long long>(start),
              static_cast<unsigned long long>(count),
              static_cast<unsigned long long>(total_elements));
        return false;
      }

      // Calculate the hyperslab start and count for each dimension
      std::vector<hsize_t> start_array(rank, 0); // Initialize with 0s
      std::vector<hsize_t> count_array = dims;   // Initialize to read full dims

      hsize_t flat_start = start;
      hsize_t flat_count = count;

      // Map 1D offset and count to multi-dimensional coordinates
      for (int i = rank - 1; i >= 0; --i) {
        hsize_t dim_size = dims[i];
        start_array[i] =
            flat_start % dim_size; // Compute offset within this dimension
        flat_start /= dim_size;

        count_array[i] = (i == rank - 1)
                             ? flat_count
                             : 1; // Set count for last dimension, 1 otherwise
        flat_count =
            (i == 0)
                ? flat_count
                : flat_count / dim_size; // Adjust for multidimensional shape
      }

      // Select the hyperslab in the file dataspace
      dataspace.selectHyperslab(H5S_SELECT_SET, count_array.data(),
                                start_array.data());

      // Define the memory dataspace for contiguous reading
      H5::DataSpace memspace(
          1, &count); // 1D space with `count` elements in memory

      // Resize data to hold the read elements
      data.resize(count);
      dataset.read(data.data(), this->getHDF5Type<T>(), memspace, dataspace);

      return true;
    } catch (const H5::Exception &err) {
      error("Failed to read dataset slice '%s': %s", datasetName.c_str(),
            err.getCDetailMsg());
      return false;
    }
  }

  template <typename T>
  bool readAttribute(const std::string &objName,
                     const std::string &attributeName, T &attributeValue) {
    try {
      H5::Group group(file.openGroup(objName));
      H5::Attribute attr = group.openAttribute(attributeName);
      H5::DataType attrType = attr.getDataType();
      attr.read(attrType, &attributeValue);
      group.close();
      attr.close();
      return true;
    } catch (H5::Exception &e) {
      return false;
    }
  }

private:
  // Flags for whether file is open or closed
  bool file_open;
  bool file_closed;

  // Template declaration
  template <typename T> H5::PredType getHDF5Type();
};

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

#endif // SERIAL_IO_H_
