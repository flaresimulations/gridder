/**
 * @file cmd_parser.hpp
 * @brief Robust command line argument parser for the gridder application
 */
#ifndef CMD_PARSER_HPP
#define CMD_PARSER_HPP

#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Structure to hold parsed command line arguments
 */
struct CommandLineArgs {
  std::string parameter_file;
  int nthreads;
  int nsnap;
  int verbosity;
  bool help_requested;

  // Constructor with defaults
  CommandLineArgs() : nthreads(1), nsnap(0), verbosity(1), help_requested(false) {}
};

/**
 * @brief Robust command line parser with validation
 */
class CommandLineParser {
public:
  /**
   * @brief Parse command line arguments with comprehensive validation
   *
   * @param argc Number of command line arguments
   * @param argv Array of command line argument strings
   * @param rank MPI rank (for error reporting)
   * @param size MPI size (for validation)
   * @return CommandLineArgs structure with parsed values
   * @throws std::runtime_error if parsing fails or validation errors occur
   */
  static CommandLineArgs parse(int argc, char *argv[], int rank = 0,
                               int size = 1) {
    CommandLineArgs args;

    // Handle help request early
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      args.help_requested = true;
      return args;
    }

    // Validate argument count
    validateArgumentCount(argc, argv[0]);

    // Parse required arguments
    args.parameter_file = parseParameterFile(argv[1]);
    args.nthreads = parseThreadCount(argv[2]);

    // Parse optional arguments (snapshot and/or verbosity)
    if (argc >= 4) {
      args.nsnap = parseSnapNumber(argv[3]);
    }
    if (argc >= 5) {
      args.verbosity = parseVerbosity(argv[4]);
    }

    // Perform additional validations
    validateInputs(args, rank, size);

    return args;
  }

  /**
   * @brief Print usage information
   *
   * @param program_name Name of the program executable
   */
  static void printUsage(const char *program_name) {
    std::cerr << "\nUsage: " << program_name
              << " <parameter_file> <nthreads> [snapshot_number] [verbosity]\n\n";
    std::cerr << "Arguments:\n";
    std::cerr
        << "  parameter_file   Path to YAML parameter configuration file\n";
    std::cerr << "  nthreads        Number of OpenMP threads (1-"
              << getMaxThreads() << ")\n";
    std::cerr << "  snapshot_number Optional snapshot number (>=0, replaces "
                 "placeholder)\n";
    std::cerr << "  verbosity       Optional verbosity level: 0=minimal, 1=rank 0 only (default), 2=all ranks\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help      Show this help message\n\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << program_name << " params.yml 8\n";
    std::cerr << "  " << program_name << " params.yml 16 42\n";
    std::cerr << "  " << program_name << " params.yml 8 0 2  # All ranks print\n";
    std::cerr << "  mpirun -n 4 " << program_name << " params.yml 8\n\n";
  }

  /**
   * @brief Print detailed error with context
   */
  static void printError(const std::string &error_msg,
                         const char *program_name) {
    std::cerr << "\nError: " << error_msg << "\n";
    printUsage(program_name);
  }

private:
  /**
   * @brief Validate the number of command line arguments
   */
  static void validateArgumentCount(int argc, const char * /* program_name */) {
    if (argc < 3 || argc > 5) {
      std::ostringstream oss;
      oss << "Invalid number of arguments (" << (argc - 1)
          << "). Expected 2-4 arguments.";
      throw std::runtime_error(oss.str());
    }
  }

  /**
   * @brief Parse and validate parameter file argument
   */
  static std::string parseParameterFile(const char *arg) {
    std::string param_file(arg);

    // Check if file exists and is readable
    if (!std::filesystem::exists(param_file)) {
      std::ostringstream oss;
      oss << "Parameter file '" << param_file << "' does not exist";
      throw std::runtime_error(oss.str());
    }

    if (!std::filesystem::is_regular_file(param_file)) {
      std::ostringstream oss;
      oss << "Parameter file '" << param_file << "' is not a regular file";
      throw std::runtime_error(oss.str());
    }

    // Check file extension (optional but helpful)
    std::filesystem::path path(param_file);
    std::string ext = path.extension().string();
    if (ext != ".yml" && ext != ".yaml") {
      std::cerr << "Warning: Parameter file '" << param_file
                << "' does not have .yml/.yaml extension\n";
    }

    // Check file permissions
    std::error_code ec;
    auto perms = std::filesystem::status(param_file, ec).permissions();
    if (ec || (perms & std::filesystem::perms::owner_read) ==
                  std::filesystem::perms::none) {
      std::ostringstream oss;
      oss << "Parameter file '" << param_file << "' is not readable";
      throw std::runtime_error(oss.str());
    }

    // Basic file size sanity check
    auto file_size = std::filesystem::file_size(param_file, ec);
    if (!ec) {
      if (file_size == 0) {
        std::ostringstream oss;
        oss << "Parameter file '" << param_file << "' is empty";
        throw std::runtime_error(oss.str());
      }
      if (file_size > 10 * 1024 * 1024) { // 10MB limit for parameter files
        std::cerr << "Warning: Parameter file '" << param_file
                  << "' is unusually large (" << (file_size / 1024) << " KB)\n";
      }
    }

    return param_file;
  }

  /**
   * @brief Parse and validate thread count
   */
  static int parseThreadCount(const char *arg) {
    int nthreads;

    // Parse integer
    try {
      size_t pos;
      nthreads = std::stoi(arg, &pos);

      // Check if entire string was consumed
      if (pos != std::strlen(arg)) {
        throw std::runtime_error(
            "Thread count contains non-numeric characters");
      }
    } catch (const std::invalid_argument &) {
      throw std::runtime_error("Thread count is not a valid integer");
    } catch (const std::out_of_range &) {
      throw std::runtime_error("Thread count is out of range");
    }

    // Validate range
    if (nthreads <= 0) {
      throw std::runtime_error("Thread count must be positive (got " +
                               std::to_string(nthreads) + ")");
    }

    const int max_threads = getMaxThreads();
    if (nthreads > max_threads) {
      std::ostringstream oss;
      oss << "Thread count " << nthreads << " exceeds recommended maximum "
          << max_threads;
      throw std::runtime_error(oss.str());
    }

    return nthreads;
  }

  /**
   * @brief Parse and validate snapshot number
   */
  static int parseSnapNumber(const char *arg) {
    int nsnap;

    // Parse integer
    try {
      size_t pos;
      nsnap = std::stoi(arg, &pos);

      // Check if entire string was consumed
      if (pos != std::strlen(arg)) {
        throw std::runtime_error(
            "Snapshot number contains non-numeric characters");
      }
    } catch (const std::invalid_argument &) {
      throw std::runtime_error("Snapshot number is not a valid integer");
    } catch (const std::out_of_range &) {
      throw std::runtime_error("Snapshot number is out of range");
    }

    // Validate range
    if (nsnap < 0) {
      throw std::runtime_error("Snapshot number must be non-negative (got " +
                               std::to_string(nsnap) + ")");
    }

    // Reasonable upper limit check
    if (nsnap > 9999) {
      std::cerr << "Warning: Snapshot number " << nsnap
                << " is unusually high\n";
    }

    return nsnap;
  }

  /**
   * @brief Parse and validate verbosity level
   */
  static int parseVerbosity(const char *arg) {
    int verbosity;

    // Parse integer
    try {
      size_t pos;
      verbosity = std::stoi(arg, &pos);

      // Check if entire string was consumed
      if (pos != std::strlen(arg)) {
        throw std::runtime_error(
            "Verbosity contains non-numeric characters");
      }
    } catch (const std::invalid_argument &) {
      throw std::runtime_error("Verbosity is not a valid integer");
    } catch (const std::out_of_range &) {
      throw std::runtime_error("Verbosity is out of range");
    }

    // Validate range (0=minimal, 1=rank 0 only, 2=all ranks)
    if (verbosity < 0 || verbosity > 2) {
      throw std::runtime_error("Verbosity must be 0, 1, or 2 (got " +
                               std::to_string(verbosity) + ")");
    }

    return verbosity;
  }

  /**
   * @brief Perform additional validation on parsed arguments
   */
  static void validateInputs(const CommandLineArgs &args, int rank, int size) {
    // MPI-specific validation
    if (size > 1) {
      // Only print MPI warnings from rank 0 to avoid spam
      if (rank == 0) {
        if (args.nthreads > 16) {
          std::cerr << "Warning: Using " << args.nthreads
                    << " threads per MPI rank may cause oversubscription\n";
        }
      }
    }

    // Resource usage warning
    const int total_threads = args.nthreads * size;
    const int hardware_threads = std::thread::hardware_concurrency();

    if (rank == 0 && hardware_threads > 0 &&
        total_threads > hardware_threads * 2) {
      std::cerr << "Warning: Total threads (" << total_threads
                << ") significantly exceeds hardware concurrency ("
                << hardware_threads << ")\n";
    }
  }

  /**
   * @brief Get reasonable maximum thread count
   */
  static int getMaxThreads() {
    const int hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads > 0 ? hardware_threads * 2
                                : 64; // Fallback to reasonable limit
  }
};

#endif // CMD_PARSER_HPP
