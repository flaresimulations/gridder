#ifndef PARAMS_H_
#define PARAMS_H_

// Standard includes
#include <cctype>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

/* Define a variant type to hold different data types */
using Param = std::variant<int, double, std::string>;

class Parameters {
public:
  Parameters() = default;

  // Prototypes for member functions (defined in params.cpp)
  template <typename T>
  void setParameter(const std::string &key, const T &value);
  template <typename T> T getParameter(const std::string &key, T defaultValue);
  template <typename T> T getParameterNoDefault(const std::string &key);
  std::string getParameterString(const std::string &key,
                                 std::string defaultValue);
  bool exists(const std::string &key);
  void printAllParameters();

private:
  /** @brief Map to store key-value pairs
   *
   *
   */
  std::map<std::string, Param> parameters;
};

// Prototypes for helper functions (defined in params.cpp)
std::string getInputFilePath(Parameters *params, const int nsnap);
std::string getOutputFilePath(Parameters *params, const int nsnap);
std::string getGridFilePath(Parameters *params, const int nsnap,
                            const std::string &grid_file);
Parameters *parseParams(const std::string &filename);

// Prototypes for helper functions (defined in params.cpp)
Param stringToVariant(const std::string &str);

/**
 * @brief Set a key-value pair for a parameter.
 *
 * @param key The key for the parameter.
 * @param value The value for the parameter.
 */
template <typename T>
void Parameters::setParameter(const std::string &key, const T &value) {
  parameters[key] = value;
}

/**
 * @brief Get a parameter from the map, or return the default value.
 *
 * @param key The key for the parameter.
 * @param defaultValue The default value for the parameter.
 */
template <typename T>
T Parameters::getParameter(const std::string &key, T defaultValue) {

  /* Get the parameter if exists, or error. */
  if (parameters.count(key) > 0) {
    return std::get<T>(parameters.at(key));
  } else {
    setParameter(key, defaultValue);
    return defaultValue;
  }
}

/**
 * @brief Get a parameter from the map, or error if it does not exist.
 *
 * @param key The key for the parameter.
 */
template <typename T>
T Parameters::getParameterNoDefault(const std::string &key) {

  /* Get the parameter if exists, or error. */
  if (parameters.count(key) > 0) {
    return std::get<T>(parameters.at(key));
  } else {
    printf("A required parameter was not set in the parameter file (%s)",
           key.c_str());
    throw std::runtime_error("Required parameter not found: " + key);
  }
}

#endif // PARAMS_H_
