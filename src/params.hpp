#ifndef PARAMS_H_
#define PARAMS_H_

// Standard includes
#include <cctype>
#include <iostream>
#include <map>
#include <sstream>
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
Parameters *parseParams(const std::string &filename);

/** @brief Helper function to convert a string to a YAMLValue
 *
 * @param str The string to convert
 *
 * @return The converted YAMLValue
 */
static Param stringToVariant(const std::string &str) {

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
  T value;
  if (parameters.count(key) > 0) {
    value = std::get<T>(parameters.at(key));
  } else {
    printf("A required parameter was not set in the parameter file (%s)",
           key.c_str());
  }
  return value;
}

#endif // PARAMS_H_
