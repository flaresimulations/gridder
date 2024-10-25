// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef PARTITION_HPP
#define PARTITION_HPP

// Standard includes
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// MPI includes
#include <mpi.h>

// Local includes
#include "cell.hpp"
#include "logger.hpp"
#include "partition.hpp"

// @brief Function to construct a peano-hilbert curve from the cells
//
// This peano-hilbert curve will be used to decompose the cells into
// a 1D array of cells that can be distributed across MPI ranks. While keeping
// spatial locality, this will allow for a more even distribution of cells
// across the MPI ranks.
void sortPeanoHilbert(std::shared_ptr<Cell> *cells) {

  // Get the metadata instance
  Metadata &metadata = Metadata::getInstance();

  // Define an array to store the peano-hilbert index
  std::vector<int64_t> peano_hilbert_index;

  // Loop over the cells and get the peano-hilbert index
  for (size_t i = 0; i < metadata.nr_cells; i++) {
    peano_hilbert_index.push_back(cells[i]->ph_ind);
  }

  // Create an index array
  std::vector<int> indices(peano_hilbert_index.size());

  // Initialize indices to 0, 1, 2, ..., n-1
  std::iota(indices.begin(), indices.end(), 0);

  // Sort the cell array by peano-hilbert index
  std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
    return peano_hilbert_index[i1] < peano_hilbert_index[i2];
  });

  // Rearrange the cells according to the sorted indices
  std::vector<std::shared_ptr<Cell>> temp = cells;
  for (size_t i = 0; i < indices.size(); ++i) {
    cells[i] = temp[indices[i]];
  }
}

// @brief Function to decompose the cells
void decomposeCells(std::shared_ptr<Cell> *cells) {

  // Get the metadata instance
  Metadata &metadata = Metadata::getInstance();

  // Unpack the MPI information
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Sort cells by peano-hilbert index
  sortPeanoHilbert(cells);

  // How many particles would each rank get if we split the cells evenly?
  int particles_per_rank = metadata.nr_dark_matter / size;

  if (rank == 0) {
    message("Aiming for %d particles per rank", particles_per_rank);
  }

  // Define a count to keep track of the number of particles assigned to this
  // rank
  size_t this_rank_particles = 0;

  // Loop over the cells and assign them to the ranks
  int assign_rank = 0;
  size_t particles_assigned = 0;
  for (size_t i = 0; i < metadata.nr_cells; i++) {

    // Assign the cell to the current rank
    cells[i]->rank = assign_rank;

    // Add the number of particles in the cell to assigned count
    particles_assigned += cells[i]->part_count;
    message("Currently assigned %d particles to rank %d", particles_assigned,
            assign_rank);

    // Add the number of particles in the cell to the count for this rank
    if (assign_rank == rank) {
      this_rank_particles += cells[i]->part_count;
    }

    // If the number of particles assigned to this rank is greater than the
    // number of particles per rank, move to the next rank
    if (particles_assigned > particles_per_rank) {
      assign_rank++;
      particles_assigned = 0;
    }
  }

  message("Will work with %d dark matter particles", this_rank_particles);
}

#endif // PARTITION_HPP
