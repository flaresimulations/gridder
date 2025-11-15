// Standard includes
#include <string>

// Local includes
#include "logger.hpp"
#include "metadata.hpp"
#include "params.hpp"

/**
 * @brief Parse and set all the metadata.
 *
 * Note that some metadata is attached elsewhere in parseParams and
 * parseCmdArgs.
 *
 * @param params The parameters for the simulation
 */
void readMetadata(Parameters *params) {

  tic();

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Get the maximum leaf count
  metadata->max_leaf_count = static_cast<size_t>(
      params->getParameter<int>("Tree/max_leaf_count", 200));

  // Get the input file path
  metadata->input_file = getInputFilePath(params, metadata->nsnap);

  // Get the output file path
  metadata->output_file = getOutputFilePath(params, metadata->nsnap);

  // Should we write out masses?
  metadata->write_masses =
      static_cast<bool>(params->getParameter<int>("Output/write_masses", 0));

  // What is the fraction of particles we will read in unused cells to reduce
  // I/O calls?
  metadata->gap_fill_fraction = static_cast<double>(
      params->getParameter<double>("Input/part_gap_fill_fraction", 0.01));

  message("Reading data from: %s", metadata->input_file.c_str());

  toc("Reading metadata");
}
