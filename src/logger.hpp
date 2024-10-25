#ifndef LOGGING_H
#define LOGGING_H

// Standard Includes
#include "metadata.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <string>

// MPI Includes
// #include <mpi.h>

// Log levels
enum LogLevel { ERROR, LOG, VERBOSE };

/**
 * @brief The Logging class provides a simple mechanism for logging messages
 * to the standard output.
 *
 * The class is designed to provide a simple interface for logging messages
 * to the standard output. The implementation ensures proper formatting of
 * messages and allows for filtering based on a specified log level.
 *
 * Errors are handled by throwing exceptions and storing information about
 * the error location. This information is then used to report the error
 * at the top of the call stack.
 *
 * At the bottom of this file, we define friendly macros for logging. These
 * macros are used throughout the codebase to log messages to the standard
 * output, rather than using the Logging class directly.
 *
 * Log levels:
 * - ERROR (0): Log only error messages. (Minimal output)
 * - LOG (1): Log regular messages. (Default)
 * - VERBOSE (2): Log verbose messages. (Maximum output)
 *
 */
class Logging {
private:
  // The log level
  LogLevel _level;

  // The rank of the process
  std::string _rank;

  // Time variables for measuring duration
  std::chrono::high_resolution_clock::time_point _tic;
  std::chrono::high_resolution_clock::time_point _toc;
  std::chrono::high_resolution_clock::time_point _start;

  // Error variables (used for throwing exceptions and reporting their location
  // at the top of the call stack)
  std::string error_message_;
  char *error_file_;
  char *error_func_;
  int error_line_;

  // The current snapshot being processed
  std::string _snapshot;

  // Private constructor to prevent direct instantiation
  Logging(LogLevel level) : _level(level), _rank("[...]") {}
  ~Logging() {}

  // Deleted move constructor and move assignment to ensure singleton
  Logging(Logging &&) = delete;            // Move constructor
  Logging &operator=(Logging &&) = delete; // Move assignment operator

public:
  /**
   * @brief Get the singleton instance of the Logging class.
   *
   * @param level The log level.
   *
   * @return The singleton instance of the Logging class.
   *
   * */
  static Logging *getInstance(LogLevel level = LOG) {
    static Logging instance(level);
    return &instance;
  }

  // Deleted copy constructor and copy assignment to prevent duplication
  Logging(const Logging &) = delete;            // Copy constructor
  Logging &operator=(const Logging &) = delete; // Copy assignment operator

  /**
   * @brief Set the rank of the process with 0-padding.
   *
   * @param rank The rank of the process.
   */
  void setRank(const int rank) {
    // Convert the rank to a string with zero-padding
    std::ostringstream oss;
    oss << std::setw(4) << std::setfill('0') << rank;

    // Store the result in _rank (assuming _rank is a string)
    _rank = oss.str();
  }

  /**
   * @breif Set the current snapshot being processed.
   *
   * @param snapshot The current snapshot being processed.
   * @param length The length of the snapshot number.
   */
  void setSnapshot(const int &snapshot, const int length = 4) {
    // Convert the snapshot number to a string with zero-padding
    std::ostringstream oss;
    oss << "[" << std::setw(length) << std::setfill('0') << snapshot << "]";

    // Store the result in _snapshot (assuming _snapshot is a string)
    _snapshot = oss.str();
  }

  /**
   * @brief Log a verbose message.
   *
   * @tparam Args Variadic template for message formatting.
   *
   * @param format The format string for the log message.
   * @param args The arguments for message formatting.
   */
  template <typename... Args>
  void v_message(const char *file, const char *func, const char *format,
                 Args... args) {
    if (_level >= 2) {
      log(file, func, format, args...);
    }
  }

  /**
   * @brief Log a regular log message.
   *
   * @tparam Args Variadic template for message formatting.
   *
   * @param format The format string for the log message.
   * @param args The arguments for message formatting.
   */
  template <typename... Args>
  void message(const char *file, const char *func, const char *format,
               Args... args) {
    if (_level >= LOG) {
      log(file, func, format, args...);
    }
  }

  template <typename... Args>
  void throw_error(const char *file, const char *func, int line,
                   const char *format, Args &&...args) {
    // Conditional compilation based on the number of arguments
    if constexpr (sizeof...(args) > 0) {
      // If there are arguments, process them with the format
      char buffer[512];
      std::snprintf(buffer, sizeof(buffer), format,
                    std::forward<Args>(args)...);
      this->error_message_ = buffer;
    } else {
      // If there are no arguments, use the format directly as the message
      this->error_message_ = format;
    }
    // Common handling for file, func, and line
    this->error_file_ = const_cast<char *>(file);
    this->error_func_ = const_cast<char *>(func);
    this->error_line_ = line;

    // Throw the error
    throw std::runtime_error(this->error_message_);
  }

  /**
   * @brief Log an error message and throw a runtime error.
   *
   * @param msg The error message format string.
   * @param args The arguments for message formatting.
   *
   * @throw std::runtime_error Thrown with the formatted error message.
   */
  void report_error() {
    std::ostringstream oss;
    oss << "[ERROR][" << getBaseFilename(this->error_file_) << "."
        << this->error_func_ << "." << this->error_line_
        << "]: " << this->error_message_;
    std::cerr << oss.str() << std::endl;
  }

  /**
   * @brief Start measuring time.
   */
  void tic() { _tic = std::chrono::high_resolution_clock::now(); }

  /**
   * @brief Stop measuring time, log the duration, and print the log message.
   *
   * @param message The message indicating the operation being measured.
   */
  void toc(const char *file, const char *func, const char *message) {
    _toc = std::chrono::high_resolution_clock::now();

    // Only rank 0 should print
    if (Metadata::getInstance().rank != 0) {
      return;
    }

    // Calculate the duration...
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic);

    // And report it...
    log(file, func, "%s took %lld ms", message,
        static_cast<long long>(duration.count()));
  }

  /**
   * @brief Start measuring time.
   */
  void start() { _start = std::chrono::high_resolution_clock::now(); }

  /**
   * @brief Report the full runtime of the program.
   */
  void finish(const char *file, const char *func) {

    // Only rank 0 should print
    if (Metadata::getInstance().rank != 0) {
      return;
    }

    // Calculate the duration...
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - _start);

    // And report it...
    log(file, func, "Total runtime: %lld ms",
        static_cast<long long>(duration.count()));
  }

private:
  /**
   * @brief Get the base filename from a given file path.
   *
   * @param filePath The full path to the file.
   * @return The base filename without the path and extension.
   */
  static std::string getBaseFilename(const std::string &filePath) {
    size_t lastSlash = filePath.find_last_of("/");
    size_t lastDot = filePath.find_last_of(".");

    // Extract the filename between the last slash and the last dot
    if (lastSlash != std::string::npos && lastDot != std::string::npos &&
        lastDot > lastSlash) {
      return filePath.substr(lastSlash + 1, lastDot - lastSlash - 1);
    }

    // If no slash or dot found, or dot appears before slash, return the
    // original path
    return filePath;
  }

  /**
   * @brief Log a formatted message.
   *
   * @tparam Args Variadic template for message formatting.
   *
   * @param format The format string for the log message.
   * @param args The arguments for message formatting.
   */
  template <typename... Args>
  void log(const char *file, const char *func, const char *format,
           Args... args) {

    // Only rank 0 should print

    // Create the standard output string format
    std::ostringstream oss;
    oss << " [" << _rank << "][" << getBaseFilename(file) << "." << func
        << "] ";

    // Format the message
    char buffer[256];
    snprintf(buffer, sizeof(buffer), format, args...);

    // Include the message
    oss << buffer << std::endl;

    // Print the message
    std::cout << oss.str();
  }

  /**
   * @brief Get the current step of the simulation.
   *
   * @return The current step of the simulation.
   */
  std::string getSnapshot() { return _snapshot; }
};

// Define friendly macros for logging
#define message(...)                                                           \
  Logging::getInstance()->message(__FILE__, __func__, __VA_ARGS__)
#define v_message(...)                                                         \
  Logging::getInstance()->v_message(__FILE__, __func__, __VA_ARGS__)
#define start() Logging::getInstance()->start()
#define tic() Logging::getInstance()->tic()
#define toc(message) Logging::getInstance()->toc(__FILE__, __func__, message)
#define finish() Logging::getInstance()->finish(__FILE__, __func__)
#define error(...)                                                             \
  Logging::getInstance()->throw_error(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define report_error() Logging::getInstance()->report_error()

#endif // LOGGING_H
