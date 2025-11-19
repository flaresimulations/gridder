/**
 * @file logger.cpp
 * @brief Implementation of logging functionality
 */

#include "logger.hpp"
#include "metadata.hpp"

bool Logging::shouldPrint() const {
  Metadata *metadata = &Metadata::getInstance();

#ifdef WITH_MPI
  // Verbosity 0: Only errors (handled elsewhere)
  // Verbosity 1: Only rank 0 prints
  // Verbosity 2: All ranks print
  if (metadata->verbosity == 0) {
    return false;
  } else if (metadata->verbosity == 1) {
    return metadata->rank == 0;
  } else {
    return true;  // verbosity >= 2
  }
#else
  // Serial mode: verbosity 0 = minimal, 1+ = print
  return metadata->verbosity > 0;
#endif
}
