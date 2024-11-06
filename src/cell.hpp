// This file is part of flares_simulations/zoom_region_selection, a C++ library
// for selecting regions from parent simulations and weight based on
// overdensity.
#ifndef CELL_HPP
#define CELL_HPP

// Standard includes
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <utility>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Local includes
#include "logger.hpp"
#include "metadata.hpp"
#include "particle.hpp"

// Forward declaration
class Grid;
class GridPoint;
class Simulation;

class Cell : public std::enable_shared_from_this<Cell> {
public:
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

#ifdef WITH_MPI
  //! What rank is this cell on?
  int rank = 0;

  //! Do we border another rank? If this is true we will need to send our
  // particles to the neighbouring rank.
  bool is_proxy;
#endif

  //! Particles within the cell
  std::vector<std::shared_ptr<Particle>> particles;

  //! Grid points within the cell
  std::vector<std::shared_ptr<GridPoint>> grid_points;

  //! Child cells
  std::array<std::shared_ptr<Cell>, 8> children;

  //! Parent cell
  std::shared_ptr<Cell> parent;

  //! Pointer to the top level cell
  std::shared_ptr<Cell> top;

  //! Store the neighbouring cells
  std::vector<std::shared_ptr<Cell>> neighbours;

  //! Depth in the tree
  int depth;

  // Constructor
  Cell(const double loc[3], const double width[3],
       std::shared_ptr<Cell> parent = nullptr,
       std::shared_ptr<Cell> top = nullptr) {

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
    this->is_proxy = false;
#endif
  }

  // Destructor
  ~Cell() {
    // Clear out all the vectors of pointers
    this->particles.clear();
    this->neighbours.clear();
    this->grid_points.clear();
  }

  // Prototypes for member functions (defined in cell.cpp)
  bool inKernel(std::shared_ptr<GridPoint> grid_point,
                const double kernel_rad2);
  bool outsideKernel(std::shared_ptr<GridPoint> grid_point,
                     const double kernel_rad2);
  void split();
};

// Prototypes for functions defined in construct_cells.cpp
void getTopCells(Simulation *sim, Grid *grid);
void splitCells(Simulation *sim);

// Prototypes for functions defined in cell.cpp
std::shared_ptr<Cell> getCellContainingPoint(const std::shared_ptr<Cell> *cells,
                                             const double pos[3]);
std::shared_ptr<Cell>
getCellIndexContainingPoint(const std::shared_ptr<Cell> *cells,
                            const double pos[3]);
void assignPartsAndPointsToCells(std::shared_ptr<Cell> *cells);

// Prototypes for functions defined in cell_search.cpp
void getKernelMasses(std::shared_ptr<Cell> *cells);

#endif // CELL_HPP
