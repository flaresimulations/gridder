/**
 * @file simulation.hpp
 * @brief The defintion of the Simulation class for holding data realated to
 * the simulation.
 *
 * This includes metadata defining the geometry of the simulation, the cells
 * themselves and the particles within them.
 */
#ifndef SIMULATION_HPP
#define SIMULATION_HPP

// Standard includes

// Local includes
#include "cell.hpp"
#include "hdf_io.hpp"
#include "metadata.hpp"

class Simulation {

public:
  //! The number of cells in the simulation
  size_t nr_cells;

  //! The number of particles in the simulation
  size_t nr_particles[6];

  //! The number of dark matter particles in the simulation
  size_t nr_dark_matter;

  //! The number of cells along an axis in the Simulation
  int cdim[3];

  //! The width of a cell
  double width[3];

  //! The width of the simulation box
  double dim[3];

  //! The maximum depth in the cell tree
  int max_depth = 0;

  //! The redshift of the snapshot
  double redshift;

  //! The comoving mean density of the universe (we define this directly from
  // the matter distribution)
  double mean_density;

  //! An array of the cells
  std::shared_ptr<Cell> *cells;

  //! The number of particles in each cell
  std::vector<int> cell_part_counts;

  //! The indices to the particles in each cell in the simulation output
  std::vector<int> cell_part_starts;

  /**
   * @brief Construct a new Simulation object
   *
   * @param nr_cells The number of cells in the simulation
   * @param nr_particles The number of particles in the simulation
   * @param cdim The number of cells along an axis in the Simulation
   * @param width The width of the simulation box
   */
  Simulation() {

    // Read the simulation data from the input file
    readSimulationData();

    // Allocate the cells array
    this->cells = new std::shared_ptr<Cell>[nr_cells];
  }

  /**
   * @brief Destroy the Simulation object
   */
  ~Simulation() {
    // Delete the cells array
    delete[] this->cells;
  }

  /**
   * @brief Read the simulation data from the input file
   */
  void readSimulationData() {
    // Get the metadata instance
    Metadata *metadata = &Metadata::getInstance();

    // Set up the HDF5 object
    HDF5Helper hdf(metadata->input_file);

    // Read the metadata from the file
    hdf.readAttribute<double>(std::string("Header"), std::string("Redshift"),
                              this->redshift);
    hdf.readAttribute<int[6]>(std::string("Header"),
                              std::string("NumPart_Total"), this->nr_particles);
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
  }
};

#endif // SIMULATION_HPP
