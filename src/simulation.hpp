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

  //! The inverse of the width of a cell
  double inv_width[3];

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

  //! The number of particles in each cell
  std::vector<size_t> cell_part_counts;

  //! The indices to the particles in each cell in the simulation output
  std::vector<size_t> cell_part_starts;

  //! Vector of pointers to locally useful cells (for efficient iteration)
  std::vector<Cell*> locally_useful_cells;

  //! Vector of pointers to all useful cells (for efficient iteration)
  std::vector<Cell*> useful_cells;

  //! Particle masses and flattened xyz positions. Cells store indices into
  //! these simulation-owned arrays rather than Particle objects.
  std::vector<double> particle_masses;
  std::vector<double> particle_positions;

  //! True once particle properties have been physically regrouped into
  //! contiguous top-cell ranges ready for SWIFT-like recursive partitioning.
  bool particle_ranges_enabled = false;

  //! Append one particle and return its stable index.
  ParticleIndex appendParticle(const double pos[3], double mass) {
    const ParticleIndex index = particle_masses.size();
    particle_positions.insert(particle_positions.end(), pos, pos + 3);
    try {
      particle_masses.push_back(mass);
    } catch (...) {
      particle_positions.resize(index * 3);
      throw;
    }
    return index;
  }

  //! Return the xyz position associated with a particle index.
  const double *particlePosition(ParticleIndex index) const {
    return &particle_positions[index * 3];
  }

  //! Return the mass associated with a particle index.
  double particleMass(ParticleIndex index) const {
    return particle_masses[index];
  }

  // Constructor prototype
  Simulation();

  // Destructor prototype
  ~Simulation();

  // Prototype for reader function (defined in simulation.cpp)
  void readSimulationData();

  // Calculate mean density from cosmological parameters
  void calculateMeanDensityFromCosmology(Parameters *params);

private:
  // Helper function for cleanup
  void deleteChildCells(Cell *cell);
};

#endif // SIMULATION_HPP
