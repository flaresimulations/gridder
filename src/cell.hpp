// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <array>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "metadata.hpp"
#include "particle.hpp"

// Forward declaration
class Grid;
class GridPoint;
class Simulation;

class Cell {
public:
  //! Constants for octree structure
  static constexpr int OCTREE_CHILDREN = 8;
  static constexpr int OCTREE_DIM = 2; // 2x2x2 = 8 children
  static constexpr int SPATIAL_DIMS = 3;
  static constexpr int MAX_OCTREE_DEPTH = 64;
  //! Cell location
  double loc[3];

  //! Cell width
  double width[3];

  //! The number of particles in the cell
  size_t part_count;

  //! The mass of the cell (i.e. the sum of particle masses). This is useful
  // for adding mass to grid points entirely inside a kernel without looping
  // over particles.
  double mass;

  //! Flag for whether the cell has been split (i.e. has children)
  bool is_split;

  //! What rank is this cell on?
  int rank = 0;

  //! Is this cell a proxy for another rank? (i.e. does it receive particles
  //! from another rank)
  bool is_proxy = false;

  //! If we are sending particles to another rank where are they going? (Can be
  //! multiple ranks)
  std::vector<int> send_ranks;

  //! Flag for whether the cell is "useful" (i.e. contains grid points or
  // is a neighbour of a useful cell)
  bool is_useful = false;

  //! Particles within the cell
  std::vector<Particle *> particles;

  //! Grid points within the cell
  std::vector<GridPoint *> grid_points;

  //! Child cells
  std::array<Cell *, OCTREE_CHILDREN> children;

  //! Parent cell
  Cell *parent;

  //! Pointer to the top level cell
  Cell *top;

  //! Store the neighbouring cells
  std::vector<Cell *> neighbours;

  //! Depth in the tree
  int depth;

#ifdef DEBUGGING_CHECKS
  //! The index of the cell in the cells array (for debugging)
  int ph_ind = -1;
#endif

  // Default constructor
  Cell() {
    // Initialize to safe default values
    this->loc[0] = this->loc[1] = this->loc[2] = 0.0;
    this->width[0] = this->width[1] = this->width[2] = 0.0;
    this->parent = nullptr;
    this->top = nullptr;
    this->part_count = 0;
    this->mass = 0.0;
    this->depth = 0;
    this->is_split = false;
    this->is_useful = false;
    for (int i = 0; i < OCTREE_CHILDREN; i++) {
      this->children[i] = nullptr;
    }
#ifdef WITH_MPI
    this->rank = 0;
#endif
  }

  // Constructor
  Cell(const double loc[3], const double width[3], Cell *parent = nullptr,
       Cell *top = nullptr) {

    // Set the location and width of the cell
    this->loc[0] = loc[0];
    this->loc[1] = loc[1];
    this->loc[2] = loc[2];
    this->width[0] = width[0];
    this->width[1] = width[1];
    this->width[2] = width[2];

    // Set the parent and top level cell
    this->parent = parent;
    this->top = top;

    // Cell is never split at initialisation
    this->is_split = false;

    // Initialise the mass and particle count
    this->mass = 0.0;
    this->part_count = 0;

    // Initialise the depth
    this->depth = parent ? parent->depth + 1 : 0;

#ifdef WITH_MPI
    // Initialise the rank and proxy flags
    this->rank = 0;
    this->send_ranks.clear();
#endif
  }

  // Destructor
  ~Cell() {
    // Clear out all the vectors of pointers
    this->particles.clear();
    this->neighbours.clear();
    this->grid_points.clear();
    this->children.fill(nullptr);
    this->parent = nullptr;
    this->top = nullptr;
#ifdef WITH_MPI
    this->send_ranks.clear();
#endif
  }

  // Prototypes for member functions (defined in cell.cpp)
  bool inKernel(const GridPoint *grid_point, const double kernel_rad2) const;
  bool outsideKernel(const GridPoint *grid_point,
                     const double kernel_rad2) const;
  void split();
  void addParticle(Particle *part);
  void addGridPoint(GridPoint *grid_point) {
    this->grid_points.push_back(grid_point);
  }
};

// Prototypes for functions defined in construct_cells.cpp
void getTopCells(Simulation *sim, Grid *grid);
void splitCells(Simulation *sim);

// Prototypes for functions defined in cell.cpp
Cell *getCellContainingPoint(const double pos[3]);
int getCellIndexContainingPoint(const double pos[3]);
void assignPartsToCells(Simulation *sim);
void checkAndMoveParticles(Simulation *sim);
void assignGridPointsToCells(Simulation *sim, Grid *grid);

// Prototypes for functions defined in cell_search.cpp
void getKernelMasses(Simulation *sim, Grid *grid);
void limitToUsefulCells(Simulation *sim);

#endif // CELL_HPP
