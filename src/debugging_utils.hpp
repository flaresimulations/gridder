#ifndef DEBUGGING_UTILS_HPP
#define DEBUGGING_UTILS_HPP

// Header for comprehensive debugging utilities
// These are active only when DEBUGGING_CHECKS is defined

#include "cell.hpp"
#include "grid_point.hpp"
#include "logger.hpp"
#include "metadata.hpp"
#include "simulation.hpp"
#include <vector>

#ifdef DEBUGGING_CHECKS

/**
 * @brief Validate that grid points are correctly assigned to their containing
 * cells
 *
 * @param sim Simulation object
 * @param grid Grid object
 */
void validateGridPointCellAssignment(Simulation *sim, Grid *grid);

/**
 * @brief Validate that all useful cells have been correctly identified
 *
 * @param sim Simulation object
 * @param grid Grid object
 */
void validateUsefulCells(Simulation *sim, Grid *grid);

/**
 * @brief Validate that particles are in the correct cells
 *
 * @param sim Simulation object
 */
void validateParticleCellAssignment(Simulation *sim);

/**
 * @brief Check if grid points can find any particles within kernel radii
 *
 * @param sim Simulation object
 * @param grid Grid object
 */
void validateGridPointsHaveParticles(Simulation *sim, Grid *grid);

/**
 * @brief Validate octree structure integrity
 *
 * @param sim Simulation object
 */
void validateOctreeStructure(Simulation *sim);

/**
 * @brief Check that file-based grid points are within simulation boundaries
 *
 * @param grid Grid object
 * @param sim Simulation object
 */
void validateFileGridPoints(Grid *grid, Simulation *sim);

/**
 * @brief Print detailed diagnostics about a specific grid point
 *
 * @param grid_point The grid point to diagnose
 * @param sim Simulation object
 * @param grid Grid object
 */
void diagnoseGridPoint(GridPoint *grid_point, Simulation *sim, Grid *grid);

/**
 * @brief Count particles within radius of a grid point (brute force)
 *
 * @param grid_point The grid point
 * @param sim Simulation object
 * @param radius Search radius
 * @return Number of particles found
 */
int bruteForceCountParticles(GridPoint *grid_point, Simulation *sim,
                             double radius);

#endif // DEBUGGING_CHECKS

#endif // DEBUGGING_UTILS_HPP
