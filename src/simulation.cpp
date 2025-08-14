// Local includes
#include "simulation.hpp"
#include "cell.hpp"
#include "hdf_io.hpp"
#include "metadata.hpp"

/**
 * @brief Construct a new Simulation object
 *
 * @param nr_cells The number of cells in the simulation
 * @param nr_particles The number of particles in the simulation
 * @param cdim The number of cells along an axis in the Simulation
 * @param width The width of the simulation box
 */
Simulation::Simulation() {

  // Read the simulation data from the input file
  this->readSimulationData();

  // Allocate the cells array
  this->cells.resize(this->nr_cells);
  
  // Reserve space for dynamic storage to avoid reallocations
  // Estimate: sub_cells could be up to nr_cells * 8 (if all split once)  
  this->sub_cells.reserve(this->nr_cells * 2);  // Conservative estimate
  
  // Estimate: particles should be around nr_dark_matter
  this->particles.reserve(this->nr_dark_matter);
}


/**
 * @brief Read the simulation data from the input file.
 */
void Simulation::readSimulationData() {
  // Get the metadata instance
  Metadata *metadata = &Metadata::getInstance();

  // Set up the HDF5 object
  HDF5Helper hdf(metadata->input_file);

  // Read the metadata from the file
  hdf.readAttribute<double>(std::string("Header"), std::string("Redshift"),
                            this->redshift);
  hdf.readAttribute<size_t[6]>(
      std::string("Header"), std::string("NumPart_Total"), this->nr_particles);
  this->nr_dark_matter = this->nr_particles[1];
  hdf.readAttribute<int[3]>(std::string("Cells/Meta-data"),
                            std::string("dimension"), this->cdim);
  hdf.readAttribute<double[3]>(std::string("Cells/Meta-data"),
                               std::string("size"), this->width);
  hdf.readAttribute<double[3]>(std::string("Header"), std::string("BoxSize"),
                               this->dim);

  // Count the cells
  this->nr_cells = this->cdim[0] * this->cdim[1] * this->cdim[2];

  // Report interesting things
  message("Redshift: %f", this->redshift);
  message("Running with %d dark matter particles", this->nr_dark_matter);
  message("Running with %d cells", this->nr_cells);
  message("Cdim: %d %d %d", this->cdim[0], this->cdim[1], this->cdim[2]);
  message("Box size: %f %f %f", this->dim[0], this->dim[1], this->dim[2]);
  message("Cell size: %f %f %f", this->width[0], this->width[1],
          this->width[2]);

  // Read the number of particles in each cell
  hdf.readDataset<int>(std::string("Cells/Counts/PartType1"),
                       this->cell_part_counts);

  // Read the start index of the particles in each cell
  hdf.readDataset<int>(std::string("Cells/OffsetsInFile/PartType1"),
                       this->cell_part_starts);

  hdf.close();

  // Compute the comoving volume of the simulation
  this->volume = this->dim[0] * this->dim[1] * this->dim[2];
}
