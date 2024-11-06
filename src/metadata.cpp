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

  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Get the maximum leaf count
  metadata->max_leaf_count =
      params->getParameter<int>("Tree/max_leaf_count", 200);

  // Get the input file path
  metadata->input_file = getInputFilePath(params, metadata->nsnap);

  // Get the output file path
  metadata->output_file = getOutputFilePath(params, metadata->nsnap);

  message("Reading data from: %s", metadata->input_file.c_str());
}
