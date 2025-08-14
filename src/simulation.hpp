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

// Local includes
#include "cell.hpp"
#include "hdf_io.hpp"
#include "metadata.hpp"
#include "particle.hpp"

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

  //! The comoving volume of the simulation
  double volume;

  //! The maximum depth in the cell tree
  int max_depth = 0;

  //! The redshift of the snapshot
  double redshift;

  //! The comoving mean density of the universe (we define this directly from
  // the matter distribution)
  double mean_density;

  //! An array of the cells
  std::vector<Cell> cells;

  //! Storage for dynamically created child cells
  std::vector<Cell> sub_cells;

  //! Storage for particles
  std::vector<Particle> particles;

  //! The number of particles in each cell
  std::vector<int> cell_part_counts;

  //! The indices to the particles in each cell in the simulation output
  std::vector<int> cell_part_starts;

  // Constructor prototype
  Simulation();

  // Prototype for reader function (defined in simulation.cpp)
  void readSimulationData();
};

#endif // SIMULATION_HPP
