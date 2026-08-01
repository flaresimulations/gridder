// Standard includes
#include <algorithm>
#include <cmath>
#include <limits>

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
  // Particle properties are owned by the simulation arrays. Cells only store
  // indices, so only the dynamically allocated child cells need deletion.
  for (Cell &cell : this->cells) {
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
    message("Running with %d cells", this->nr_cells);
    message("Cdim: %d %d %d", this->cdim[0], this->cdim[1], this->cdim[2]);
    message("Box size: %f %f %f", this->dim[0], this->dim[1], this->dim[2]);
    message("Cell size: %f %f %f", this->width[0], this->width[1],
            this->width[2]);
  }

  const std::vector<hsize_t> mass_dims =
      hdf.getDatasetDimensions("PartType1/Masses");
  const std::vector<hsize_t> position_dims =
      hdf.getDatasetDimensions("PartType1/Coordinates");

  if (mass_dims.size() != 1) {
    error("PartType1/Masses must be one-dimensional");
  }
  if (position_dims.size() != 2 || position_dims[1] != 3) {
    error("PartType1/Coordinates must have shape (N, 3)");
  }
  if (mass_dims[0] != position_dims[0]) {
    error("Particle dataset size mismatch: Masses has %llu rows but "
          "Coordinates has %llu",
          static_cast<unsigned long long>(mass_dims[0]),
          static_cast<unsigned long long>(position_dims[0]));
  }
  if (mass_dims[0] > std::numeric_limits<size_t>::max()) {
    error("Particle dataset contains too many rows for this platform");
  }

  const size_t dataset_particle_count = static_cast<size_t>(mass_dims[0]);
  if (this->nr_dark_matter != dataset_particle_count) {
    message("Warning: Header particle count (%zu) does not match PartType1 "
            "datasets (%zu); using the dataset dimensions",
            this->nr_dark_matter, dataset_particle_count);
    this->nr_dark_matter = dataset_particle_count;
    this->nr_particles[1] = dataset_particle_count;
  }
  if (metadata->rank == 0)
    message("Running with %zu dark matter particles", this->nr_dark_matter);

  // Read the number of particles and 64-bit-safe starting offset for each cell.
  hdf.readDataset<size_t>(std::string("Cells/Counts/PartType1"),
                          this->cell_part_counts);

  // Read the start index of the particles in each cell
  hdf.readDataset<size_t>(std::string("Cells/OffsetsInFile/PartType1"),
                          this->cell_part_starts);

  const bool offsets_have_end_sentinel =
      this->cell_part_starts.size() == this->nr_cells + 1;
  if (this->cell_part_counts.size() != this->nr_cells ||
      (this->cell_part_starts.size() != this->nr_cells &&
       !offsets_have_end_sentinel)) {
    error("Cell metadata size mismatch: expected %zu counts and %zu or %zu "
          "offsets, found %zu counts and %zu offsets",
          this->nr_cells, this->nr_cells, this->nr_cells + 1,
          this->cell_part_counts.size(), this->cell_part_starts.size());
  }

  struct ParticleRange {
    size_t offset;
    size_t count;
    size_t cell_id;
  };

  std::vector<ParticleRange> particle_ranges;
  particle_ranges.reserve(this->nr_cells);
  size_t total_cell_particles = 0;
  for (size_t cid = 0; cid < this->nr_cells; cid++) {
    const size_t offset = this->cell_part_starts[cid];
    const size_t count = this->cell_part_counts[cid];

    if (offset > dataset_particle_count ||
        count > dataset_particle_count - offset) {
      error("Particle range for cell %zu exceeds dataset: offset=%zu, "
            "count=%zu, particles=%zu",
            cid, offset, count, dataset_particle_count);
    }
    if (count > dataset_particle_count - total_cell_particles) {
      error("Cell particle counts exceed dataset size at cell %zu", cid);
    }
    total_cell_particles += count;
    if (count > 0)
      particle_ranges.push_back({offset, count, cid});
  }

  if (total_cell_particles != dataset_particle_count) {
    error("Cell particle counts sum to %zu but datasets contain %zu particles",
          total_cell_particles, dataset_particle_count);
  }

  // Cell IDs need not follow particle-file order. Sort non-empty ranges by
  // their file offsets, then prove that they cover the datasets exactly once.
  std::sort(particle_ranges.begin(), particle_ranges.end(),
            [](const ParticleRange &left, const ParticleRange &right) {
              return left.offset < right.offset;
            });

  size_t expected_offset = 0;
  for (const ParticleRange &range : particle_ranges) {
    if (range.offset != expected_offset) {
      error("Particle ranges overlap or leave a gap before cell %zu: expected "
            "offset %zu, found %zu",
            range.cell_id, expected_offset, range.offset);
    }
    expected_offset += range.count;
  }
  if (offsets_have_end_sentinel &&
      this->cell_part_starts.back() != dataset_particle_count) {
    error("Final cell offset is %zu but datasets contain %zu particles",
          this->cell_part_starts.back(), dataset_particle_count);
  }

  // Internally offsets are one-per-cell; discard an optional exclusive end.
  this->cell_part_starts.resize(this->nr_cells);

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
  // G in (10^10 Msun)^-1 Mpc (km/s)^2 internal units
  // Derived from G_SI = 6.674e-11 m^3 kg^-1 s^-2 with proper unit conversion
  const double G = 4.301744232015554e+01; // Gravitational constant in (10^10 Msun)^-1 Mpc (km/s)^2

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
    message("Mean comoving density at z=%.4f: %.6e 10^10 Msun/cMpc^3",
            this->redshift, this->mean_density);
  }
}
