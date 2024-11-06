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
 * This is header file contains prototypes related to parsing and using the
 * parameter file.
 ******************************************************************************/
#ifndef PARAMS_H_
#define PARAMS_H_

// Standard includes
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <variant>

// Local includes
#include "logger.hpp"

/* Define a variant type to hold different data types */
using Param = std::variant<int, double, std::string>;

/** @brief Helper function to convert a string to a YAMLValue
 *
 */
Param stringToVariant(const std::string &str) {

  /* Set up conversion varibales. */
  int intValue;
  double doubleValue;
  std::istringstream intStream(str);
  std::istringstream doubleStream(str);

  /* Does it contain a decimal place? */
  bool isString = false;
  int decimalCount = 0;
  for (char c : str) {

    /* Check if character is a decimal point, if not check its not a
     * digit. If its not a digit we know we have a string. */
    if (c == '.') {
      decimalCount++;
    } else if (!(isdigit(c))) {
      isString = true;
      break;
    }
  }

  /* Return if its a string. */
  if (isString) {
    // But before we return just strip off any quotes.
    if (str.front() == '"' && str.back() == '"') {
      return str.substr(1, str.length() - 2);
    }
    return str;
  }

  /* If it isn't a string and it only has one deminal point its a double. */
  if (decimalCount == 1) {
    doubleStream >> doubleValue;
    return doubleValue;
  }

  /* Test if it's an integer */
  else if (intStream >> intValue) {

    /* It is! */
    return intValue;

  }

  /* Otherwise, something bizzare has happened... */
  else {
    printf("Parameter %s could not be converted to string, double, or int!",
           str.c_str());
    return str;
  }
}

class Parameters {
public:
  /** @brief
   *
   */
  Parameters() = default;

  /** @brief
   *
   */
  template <typename T>
  void setParameter(const std::string &key, const T &value) {
    parameters[key] = value;
  }

  /** @brief
   *
   */
  template <typename T> T getParameter(const std::string &key, T defaultValue) {

    /* Get the parameter if exists, or error. */
    if (parameters.count(key) > 0) {
      return std::get<T>(parameters.at(key));
    } else {
      setParameter(key, defaultValue);
      return defaultValue;
    }
  }

  /** @brief
   *
   */
  template <typename T> T getParameterNoDefault(const std::string &key) {

    /* Get the parameter if exists, or error. */
    T value;
    if (parameters.count(key) > 0) {
      value = std::get<T>(parameters.at(key));
    } else {
      printf("A required parameter was not set in the parameter file (%s)",
             key.c_str());
    }
    return value;
  }

  /** @brief
   *
   */
  std::string getParameterString(const std::string &key,
                                 std::string defaultValue) {

    /* Get the parameter if exists, or error. */
    if (parameters.count(key) > 0) {
      return std::get<std::string>(parameters.at(key));
    } else {
      setParameter(key, defaultValue);
      return defaultValue;
    }
  }

  /**
   * @brief Fucntion to check if a parameter exists.
   */
  bool exists(const std::string &key) { return parameters.count(key) > 0; }

  /** @brief Function to print all key-value pairs stored in the map.
   *
   */
  void printAllParameters() const {
    std::cout << "Key-Value Pairs:" << std::endl;
    for (const auto &pair : parameters) {
      std::cout << "Key: " << pair.first << " - Value: ";
      const Param &value = pair.second;

      // Print the value based on its type
      if (std::holds_alternative<int>(value)) {
        std::cout << std::get<int>(value);
      } else if (std::holds_alternative<double>(value)) {
        std::cout << std::get<double>(value);
      } else if (std::holds_alternative<std::string>(value)) {
        std::cout << std::get<std::string>(value);
      }

      std::cout << std::endl;
    }
  }

private:
  /** @brief Map to store key-value pairs
   *
   *
   */
  std::map<std::string, Param> parameters;
};

std::string getInputFilePath(Parameters *params, const int nsnap) {

  // Get the input file path and snapshot placeholder
  std::string input_file =
      params->getParameterNoDefault<std::string>("Input/filepath");
  std::string placeholder =
      params->getParameter<std::string>("Input/placeholder", "0000");

  // Ensure the placeholder is not empty to prevent infinite loops
  if (placeholder.empty()) {
    throw std::runtime_error("Placeholder cannot be an empty string.");
  }

  // Calculate the padding width based on the placeholder length
  size_t padding_width = placeholder.length();

  // Create a zero-padded string for the snapshot number
  std::ostringstream ss;
  ss << std::setw(padding_width) << std::setfill('0') << nsnap;
  std::string snap_num_str = ss.str();

  // Replace all occurrences of the placeholder with the zero-padded snapshot
  // number
  size_t pos = 0;
  while ((pos = input_file.find(placeholder, pos)) != std::string::npos) {
    input_file.replace(pos, placeholder.length(), snap_num_str);
    pos += snap_num_str.length(); // Move past the replaced segment
  }

  return input_file;
}

std::string getOutputFilePath(Parameters *params, const int nsnap) {

  // Get the output file path
  std::string output_file =
      params->getParameterNoDefault<std::string>("Output/filepath");

  // Ensure the output file path exists, if not create it
  if (!std::filesystem::exists(output_file)) {
    std::filesystem::create_directories(output_file);
  }

  // Ensure the output file path ends with a forward slash
  if (output_file.back() != '/') {
    output_file += "/";
  }

  // Combine the file path and the basename for the output file
  output_file += params->getParameterNoDefault<std::string>("Output/basename");

  // Next we need to replace any placeholders in the output file path
  // which represent the snapshot number

  // Get the placeholder for the snapshot number
  std::string placeholder =
      params->getParameter<std::string>("Input/placeholder", "0000");

  // Ensure the placeholder is not empty to prevent infinite loops
  if (placeholder.empty()) {
    throw std::runtime_error("Placeholder cannot be an empty string.");
  }

  // Calculate the padding width based on the placeholder length
  size_t padding_width = placeholder.length();

  // Create a zero-padded string for the snapshot number
  std::ostringstream ss;
  ss << std::setw(padding_width) << std::setfill('0') << nsnap;
  std::string snap_num_str = ss.str();

  // Replace all occurrences of the placeholder with the zero-padded snapshot
  // number
  size_t pos = 0;
  while ((pos = output_file.find(placeholder, pos)) != std::string::npos) {
    output_file.replace(pos, placeholder.length(), snap_num_str);
    pos += snap_num_str.length(); // Move past the replaced segment
  }

  return output_file;
}

/** @brief Function to parse a YAML file and populate the object.
 *
 */
Parameters *parseParams(const std::string &filename) {

  // Create the parameters object
  Parameters *params = new Parameters();

  /* Open the YAML file */
  std::ifstream file(filename);
  if (!file.is_open()) {
    error("Failed to open YAML file (%s).", filename.c_str());
  }

  /* Set up some variables we'll need in the loop. */
  std::string line;
  std::string parentKey;

  /* Loop until we find the end of the file. */
  while (std::getline(file, line)) {

    /* Remove comments (any text starting with #) */
    size_t commentPos = line.find('#');
    if (commentPos != std::string::npos) {
      line.erase(commentPos);
    }

    /* Ignore empty lines. */
    if (line.empty()) {
      continue;
    }

    /* Check if the line contains a key-value pair */
    size_t colonPos = line.find(':');

    /* Extract the key */
    std::string key = line.substr(0, colonPos);

    /* Trim leading and trailing whitespace */
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);

    /* If the colon is at the end then we have a new parent key. */
    if (colonPos == key.length()) {

      parentKey = key;

    } else if (colonPos != std::string::npos) {

      /* Get the value. */
      std::string valueStr = line.substr(colonPos + 1);

      /* Trim leading and trailing whitespace */
      valueStr.erase(0, valueStr.find_first_not_of(" \t"));
      valueStr.erase(valueStr.find_last_not_of(" \t") + 1);

      /* Convert the value string to a variant containing the correct
       * data type. */
      Param value = stringToVariant(valueStr);

      /* Store the key-value pair using the right data type. */
      if (std::holds_alternative<int>(value)) {
        params->setParameter(parentKey + "/" + key, std::get<int>(value));
      } else if (std::holds_alternative<double>(value)) {
        params->setParameter(parentKey + "/" + key, std::get<double>(value));
      } else if (std::holds_alternative<std::string>(value)) {
        params->setParameter(parentKey + "/" + key,
                             std::get<std::string>(value));
      }
    }
  }

  return params;
}

#endif // PARAMS_H_
