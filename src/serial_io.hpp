/*******************************************************************************
 * This file is part of MEGA++.
 * Copyright (c) 2023 Will Roper (w.roper@sussex.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This header file contains the prototypes for reading from and writing to
 * a HDF5 file in serial.
 ******************************************************************************/
#ifndef SERIAL_IO_H_
#define SERIAL_IO_H_

// Standard includes
#include <H5Cpp.h>
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

  bool writeAttribute(const std::string &objName,
                      const std::string &attributeName,
                      const std::string &attributeValue) {
    try {
      H5::Group group(file.openGroup(objName));
      H5::DataSpace dataspace(H5S_SCALAR);
      H5::StrType strType(H5::PredType::C_S1, attributeValue.length());
      H5::Attribute attr =
          group.createAttribute(attributeName, strType, dataspace);
      attr.write(strType, attributeValue);
      group.close();
      dataspace.close();
      attr.close();
      return true;
    } catch (H5::Exception &e) {
      return false;
    }
  }

  template <typename T>
  bool writeDataset(const std::string &datasetName,
                    const std::vector<T> &data) {
    try {
      H5::DataSpace dataspace(1, data.size());
      H5::DataSet dataset =
          file.createDataSet(datasetName, getHDF5Type<T>(), dataspace);
      dataset.write(data.data(), getHDF5Type<T>());
      dataspace.close();
      dataset.close();
      return true;
    } catch (H5::Exception &e) {
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
      // Open the dataset
      H5::DataSet dataset = this->file.openDataSet(datasetName);

      // Get the dataspace of the dataset
      H5::DataSpace dataspace = dataset.getSpace();

      // Get the rank (number of dimensions) of the dataspace
      int rank = dataspace.getSimpleExtentNdims();
      std::vector<hsize_t> dims(rank);
      dataspace.getSimpleExtentDims(dims.data(), NULL);

      // Prepare arrays for hyperslab parameters
      std::vector<hsize_t> start_array(rank, 0); // Initialize start positions
      std::vector<hsize_t> count_array(rank, 1); // Initialize counts

      if (rank == 1) {
        // For a 1D array
        start_array[0] = start;
        count_array[0] = count;
      } else if (rank == 2 && dims[1] == 3) {
        // For a 2D array with shape (N, 3)
        start_array[0] = start; // Start from the 'start'th row
        start_array[1] = 0;     // Start from the first column
        count_array[0] = count; // Read 'count' rows
        count_array[1] = 3;     // Read all 3 columns
      } else {
        error("Unsupported dataset rank or dimensions (rank=%d).", rank);
        return false;
      }

      // Select the hyperslab
      dataspace.selectHyperslab(H5S_SELECT_SET, count_array.data(),
                                start_array.data(), NULL, NULL);

      // Define the memory dataspace
      if (rank == 2 && dims[1] == 3) {
        // Adjust buffer size for 2D arrays to hold all elements
        std::vector<hsize_t> mem_dims = {count, 3};
        H5::DataSpace memspace(2, mem_dims.data());
        std::vector<T> buffer(count * 3); // Adjust the buffer size
        dataset.read(buffer.data(), this->getHDF5Type<T>(), memspace,
                     dataspace);
        data = std::move(buffer);
      } else {
        // Proceed as before for 1D arrays
        H5::DataSpace memspace(rank, count_array.data());
        std::vector<T> buffer(count); // Adjust the buffer size
        dataset.read(buffer.data(), this->getHDF5Type<T>(), memspace,
                     dataspace);
        data = std::move(buffer);
      }

      return true;
    } catch (const H5::Exception &err) {
      error(err.getCDetailMsg());
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
#endif // SERIAL_IO_H_
