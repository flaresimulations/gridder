// Standard includes
#include <cmath>

// Local includes
#include "simulation.hpp"
#include "cell.hpp"
#include "hdf_io.hpp"
#include "metadata.hpp"

// Define M_PI if not available (POSIX extension, not standard C++)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
}

/**
 * @brief Destructor - clean up dynamically allocated particles and cells.
 */
Simulation::~Simulation() {
  // Delete all particles allocated with raw pointers
  for (Cell &cell : this->cells) {
    for (Particle *part : cell.particles) {
      delete part;
    }
    // Recursively delete child cells (they will handle their own particles)
    deleteChildCells(&cell);
  }
}

/**
 * @brief Recursively delete all child cells.
 */
void Simulation::deleteChildCells(Cell *cell) {
  if (cell->is_split) {
    for (int i = 0; i < Cell::OCTREE_CHILDREN; i++) {
      if (cell->children[i] != nullptr) {
        deleteChildCells(cell->children[i]);
        delete cell->children[i];
        cell->children[i] = nullptr;
      }
    }
  }
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

  // Compute the inverse width of the cells
  for (int i = 0; i < 3; i++) {
    this->inv_width[i] = 1.0 / this->width[i];
  }

  // Count the cells
  this->nr_cells = this->cdim[0] * this->cdim[1] * this->cdim[2];

  // Report interesting things but only on rank 0
  if (metadata->rank == 0) {
    message("Redshift: %f", this->redshift);
    message("Running with %d dark matter particles", this->nr_dark_matter);
    message("Running with %d cells", this->nr_cells);
    message("Cdim: %d %d %d", this->cdim[0], this->cdim[1], this->cdim[2]);
    message("Box size: %f %f %f", this->dim[0], this->dim[1], this->dim[2]);
    message("Cell size: %f %f %f", this->width[0], this->width[1],
            this->width[2]);
  }

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

/**
 * @brief Calculate mean comoving density from cosmological parameters
 *
 * Computes the mean matter density at the simulation redshift using:
 * ρ_mean = ρ_crit(z=0) × Ω_m × (1+z)³
 *
 * where ρ_crit(z=0) = 3H₀²/(8πG) is the critical density today
 *
 * @param params The parameters object containing cosmology
 */
void Simulation::calculateMeanDensityFromCosmology(Parameters *params) {

  // Read cosmology parameters
  double h = params->getParameterNoDefault<double>("Cosmology/h");
  double Omega_cdm = params->getParameterNoDefault<double>("Cosmology/Omega_cdm");
  double Omega_b = params->getParameterNoDefault<double>("Cosmology/Omega_b");

  // Total matter density parameter
  double Omega_m = Omega_cdm + Omega_b;

  // Physical constants in internal units (10^10 Msun, Mpc, km/s)
  // H0 = 100 h km/s/Mpc
  double H0_kmsMpc = 100.0 * h;  // km/s/Mpc

  // Convert to internal time units (H0 in units of 1/time where time is Mpc/(km/s))
  // H0 = 100 h km/s/Mpc = 100 h / Mpc * (km/s)
  // In our units: [H0] = km/s/Mpc

  // Critical density today: ρ_crit = 3H₀²/(8πG)
  // G = 4.3009e-6 (10^10 Msun)^-1 Mpc (km/s)^2 in internal units
  const double G = 4.300917270069976e-6; // Gravitational constant in (10^10 Msun)^-1 Mpc (km/s)^2

  // ρ_crit(z=0) = 3H₀²/(8πG) in units of 10^10 Msun / Mpc^3
  double rho_crit_0 = (3.0 * H0_kmsMpc * H0_kmsMpc) / (8.0 * M_PI * G);

  // Mean COMOVING density: ρ_comoving = ρ_crit(0) × Ω_m
  // Note: In comoving coordinates, density does NOT evolve with redshift
  // The (1+z)³ factor would convert to physical density, but SWIFT uses comoving coordinates
  this->mean_density = rho_crit_0 * Omega_m;

  Metadata *metadata = &Metadata::getInstance();
  if (metadata->rank == 0) {
    message("Cosmology: h=%.4f, Omega_m=%.6f (Omega_cdm=%.6f + Omega_b=%.6f)",
            h, Omega_m, Omega_cdm, Omega_b);
    message("Critical density today: %.6e 10^10 Msun/Mpc^3", rho_crit_0);
    message("Mean comoving density (constant with z): %.6e 10^10 Msun/cMpc^3",
            this->mean_density);
    message("(Snapshot at z=%.4f, but density in comoving coordinates)", this->redshift);
  }
}
