
// Standard includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

// Local includes
#include "cell.hpp"
#include "grid_point.hpp"
#include "simulation.hpp"

namespace {
struct ParticleIndexStorage {
  size_t entries = 0;
  size_t capacity = 0;
  size_t internal_entries = 0;
};

void accumulateParticleIndexStorage(const Cell *cell,
                                    ParticleIndexStorage &storage) {
  storage.entries += cell->particles.size();
  storage.capacity += cell->particles.capacity();
  if (cell->is_split) {
    storage.internal_entries += cell->particles.size();
    for (const Cell *child : cell->children)
      accumulateParticleIndexStorage(child, storage);
  }
}

bool prepareContiguousParticleRanges(Simulation *sim) {
#ifdef WITH_MPI
  // Keep the existing indexed representation in MPI mode for now. MPI ranks
  // can retain sent particles that are no longer referenced by a local cell,
  // so their property arrays are not necessarily a live-particle permutation.
  (void)sim;
  return false;
#else
  const size_t particle_count = sim->particle_masses.size();
  if (sim->particle_positions.size() != particle_count * 3)
    error("Particle property array size mismatch before physical regrouping");

  size_t live_particle_count = 0;
  for (const Cell &cell : sim->cells) {
    if (cell.particles.size() != cell.part_count) {
      message("Cannot enable contiguous particle ranges: cell particle "
              "metadata is inconsistent");
      return false;
    }
    live_particle_count += cell.particles.size();
  }

  // A permutation can only reorder the arrays in place when every stored
  // property record is represented exactly once. Fall back for runs where
  // cleanup intentionally discarded particles from non-useful cells.
  if (live_particle_count != particle_count) {
    message("Keeping indexed particle storage: %zu live cell entries for %zu "
            "property records",
            live_particle_count, particle_count);
    return false;
  }

  const auto regroup_start = std::chrono::high_resolution_clock::now();

  // Keep top-cell chunks close to their original file ordering so the global
  // permutation is near identity when only a small fraction was misplaced.
  std::vector<size_t> cell_order(sim->nr_cells);
  std::iota(cell_order.begin(), cell_order.end(), size_t{0});
  std::stable_sort(cell_order.begin(), cell_order.end(),
                   [&](const size_t lhs, const size_t rhs) {
                     return sim->cell_part_starts[lhs] <
                            sim->cell_part_starts[rhs];
                   });

  // permutation[new_index] = old_index. Release each top-cell vector as soon
  // as its membership has been copied to limit the temporary memory overlap.
  std::vector<ParticleIndex> permutation;
  permutation.reserve(particle_count);
  size_t output_offset = 0;
  for (size_t cid : cell_order) {
    Cell &cell = sim->cells[cid];
    cell.particle_offset = output_offset;
    permutation.insert(permutation.end(), cell.particles.begin(),
                       cell.particles.end());
    output_offset += cell.particles.size();
    std::vector<ParticleIndex>().swap(cell.particles);
  }

  if (output_offset != particle_count)
    error("Failed to construct a complete top-cell particle permutation");

  // Validate bounds in all builds. Debug builds additionally prove that every
  // old particle index appears exactly once, using the high bit of permutation
  // entries as a temporary seen bit without another large allocation.
  for (ParticleIndex index : permutation) {
    if (index >= particle_count)
      error("Particle permutation contains out-of-range index %zu", index);
  }

#ifdef DEBUGGING_CHECKS
  constexpr size_t marker = size_t{1}
                            << (std::numeric_limits<size_t>::digits - 1);
  constexpr size_t value_mask = ~marker;
  if (particle_count >= marker)
    error("Particle count is too large for in-place permutation validation");

  for (size_t i = 0; i < particle_count; i++) {
    const size_t old_index = permutation[i] & value_mask;
    if (old_index >= particle_count)
      error("Particle permutation contains out-of-range index %zu", old_index);
    if ((permutation[old_index] & marker) != 0)
      error("Particle permutation contains duplicate index %zu", old_index);
    permutation[old_index] |= marker;
  }
  for (ParticleIndex &index : permutation)
    index &= value_mask;
#endif

  // Apply new[i] = old[permutation[i]] in place using disjoint permutation
  // cycles. Positions and masses always move together.
  const auto physical_reorder_start =
      std::chrono::high_resolution_clock::now();
  size_t reordered_particles = 0;
  for (size_t start = 0; start < particle_count; start++) {
    if (permutation[start] == start)
      continue;

    const double saved_pos[3] = {
        sim->particle_positions[start * 3],
        sim->particle_positions[start * 3 + 1],
        sim->particle_positions[start * 3 + 2]};
    const double saved_mass = sim->particle_masses[start];

    size_t current = start;
    while (true) {
      const size_t source = permutation[current];
      if (source == start) {
        sim->particle_positions[current * 3] = saved_pos[0];
        sim->particle_positions[current * 3 + 1] = saved_pos[1];
        sim->particle_positions[current * 3 + 2] = saved_pos[2];
        sim->particle_masses[current] = saved_mass;
        permutation[current] = current;
        reordered_particles++;
        break;
      }

      sim->particle_positions[current * 3] =
          sim->particle_positions[source * 3];
      sim->particle_positions[current * 3 + 1] =
          sim->particle_positions[source * 3 + 1];
      sim->particle_positions[current * 3 + 2] =
          sim->particle_positions[source * 3 + 2];
      sim->particle_masses[current] = sim->particle_masses[source];
      permutation[current] = current;
      current = source;
      reordered_particles++;
    }
  }

  std::vector<ParticleIndex>().swap(permutation);
  sim->particle_ranges_enabled = true;

  const double regroup_seconds =
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - regroup_start)
          .count();
  const double physical_reorder_seconds =
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - physical_reorder_start)
          .count();
  message("Physically regrouped %zu particle records into contiguous top-cell "
          "ranges in %.3f s (physical reorder %.3f s; %zu records moved)",
          particle_count, regroup_seconds, physical_reorder_seconds,
          reordered_particles);
  return true;
#endif
}
} // namespace

void getTopCells(Simulation *sim, Grid *grid) {

  tic();

  // Unpack the simulation information we need
  std::vector<Cell> &cells = sim->cells;
  const double width[3] = {sim->width[0], sim->width[1], sim->width[2]};
  const size_t nr_cells = sim->nr_cells;
  const int cdim[3] = {sim->cdim[0], sim->cdim[1], sim->cdim[2]};
  const std::vector<size_t> &counts = sim->cell_part_counts;

// Loop over the cells and create them, storing the counts for domain
// decomposition
#pragma omp parallel for
  for (size_t cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell location and width
    double loc[3] = {i * width[0], j * width[1], k * width[2]};

    // Initialize the cell in-place
    cells[cid] = Cell(loc, width, /*parent*/ nullptr);

    // Assign the particle count in this cell
    cells[cid].part_count = counts[cid];

    // We need to set top outside the constructor
    cells[cid].top = &cells[cid];
  }

  // Now the top level cells are made we can attached the pointers to
  // neighbouring cells (this simplifies boilerplate elsewhere)

  // How many cells do we need to walk out for the biggest kernel? This is
  // the maximum distance at which we will need to consider another cell.
  // We compute this separately for each dimension since cell widths and
  // grid dimensions may differ.
  int nwalk[3];
  for (int dim = 0; dim < 3; dim++) {
    nwalk[dim] = std::ceil(grid->max_kernel_radius / width[dim]) + 1;

    // Clamp to half the grid dimension to prevent duplicate neighbors
    // through periodic wrapping. If we walk more than cdim/2, we'll
    // encounter the same cell from multiple periodic images.
    if (nwalk[dim] > cdim[dim] / 2) {
      nwalk[dim] = cdim[dim] / 2;
    }
  }

  message("Looking for neighbours within [%d, %d, %d] cells",
          nwalk[0], nwalk[1], nwalk[2]);

  // Calculate maximum neighbors (use the maximum nwalk for reservation)
  const int max_nwalk = std::max({nwalk[0], nwalk[1], nwalk[2]});
  const int max_neighbors =
      (2 * max_nwalk + 1) * (2 * max_nwalk + 1) * (2 * max_nwalk + 1) -
      1; // -1 excludes self

  // Loop over the cells attaching the pointers the neighbouring cells (taking
  // into account periodic boundary conditions)
#pragma omp parallel for
  for (size_t cid = 0; cid < nr_cells; cid++) {

    // Get integer coordinates of the cell
    int i = cid / (cdim[1] * cdim[2]);
    int j = (cid / cdim[2]) % cdim[1];
    int k = cid % cdim[2];

    // Get the cell
    Cell *cell = &cells[cid];

    // Reserve space for neighbors
    cell->neighbours.reserve(max_neighbors);

    // Loop over the neighbours using dimension-specific nwalk values
    // The nwalk values are already clamped to cdim/2, preventing duplicates
    for (int ii = -nwalk[0]; ii <= nwalk[0]; ii++) {
      for (int jj = -nwalk[1]; jj <= nwalk[1]; jj++) {
        for (int kk = -nwalk[2]; kk <= nwalk[2]; kk++) {

          // Skip the cell itself
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          // Get the neighbour index (handling periodic boundary conditions)
          int iii = (i + ii + cdim[0]) % cdim[0];
          int jjj = (j + jj + cdim[1]) % cdim[1];
          int kkk = (k + kk + cdim[2]) % cdim[2];
          int cjd = iii * cdim[1] * cdim[2] + jjj * cdim[2] + kkk;

          // Skip if this wraps back to the cell itself
          // (can happen with periodic boundaries in small boxes)
          if (cjd == static_cast<int>(cid))
            continue;

          // Attach the neighbour to the cell
          cell->neighbours.push_back(&cells[cjd]);
        }
      }
    }
  }

  toc("Creating top level cells");
}

/**
 * @brief Split the top level cells to create the cell tree.
 *
 * @param cells The top level cells
 */
void splitCells(Simulation *sim) {

  prepareContiguousParticleRanges(sim);

  tic();

#ifdef WITH_MPI
  // Get the metadata instance for MPI rank checking
  Metadata *metadata = &Metadata::getInstance();

  // In MPI mode, use the locally_useful_cells lookup vector
  // Only split cells on this rank
  std::vector<Cell *> cells_to_split;
  cells_to_split.reserve(sim->locally_useful_cells.size());

  for (Cell *cell : sim->locally_useful_cells) {
    if (cell->rank == metadata->rank) {
      cells_to_split.push_back(cell);
    }
  }

  // Loop over locally useful cells on this rank and split them
#pragma omp parallel for
  for (size_t i = 0; i < cells_to_split.size(); i++) {
    cells_to_split[i]->split();
  }

  message("Rank %d: Split %zu locally useful cells", metadata->rank,
          cells_to_split.size());
#else
  // In serial mode, use the useful_cells lookup vector
  // Loop over useful cells and split them
#pragma omp parallel for
  for (size_t i = 0; i < sim->useful_cells.size(); i++) {
    sim->useful_cells[i]->split();
  }

  message("Split %zu useful cells", sim->useful_cells.size());
#endif

  if (sim->particle_ranges_enabled) {
    message("Particle tree uses contiguous property ranges with no retained "
            "particle-index storage");
  } else {
    ParticleIndexStorage index_storage;
    for (const Cell &cell : sim->cells)
      accumulateParticleIndexStorage(&cell, index_storage);

    const double logical_gib =
        static_cast<double>(index_storage.entries * sizeof(ParticleIndex)) /
        (1024.0 * 1024.0 * 1024.0);
    const double capacity_gib =
        static_cast<double>(index_storage.capacity * sizeof(ParticleIndex)) /
        (1024.0 * 1024.0 * 1024.0);
    message("Retained %zu particle indices after splitting (capacity=%zu; %.2f "
            "GiB logical, %.2f GiB capacity; %zu in internal cells)",
            index_storage.entries, index_storage.capacity, logical_gib,
            capacity_gib, index_storage.internal_entries);
  }

  message("Maximum depth in the tree: %d", sim->max_depth);

  toc("Splitting cells");
}
