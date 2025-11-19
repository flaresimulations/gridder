# FLARES-2 Gridder

High-performance cosmological simulation gridding with spherical top hat kernels.

## Overview

Computes overdensities from cosmological simulations at grid points using spherical top hat kernels. Reads HDF5 snapshots (primarily SWIFT) and applies multiple kernel radii simultaneously.

### Features

- Multiple kernel radii in single pass
- OpenMP threading + optional MPI
- Uniform, random, and file-based grids
- Octree spatial indexing
- Chunked HDF5 I/O (automatic read optimization)

### Usage

```bash
# Single-node with OpenMP
./build/parent_gridder params.yml 8

# Multi-node with MPI
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```

## Documentation

- [Quickstart](quickstart.md) - Get started quickly
- [Installation](installation.md) - Build instructions
- [Parameters](parameters.md) - Parameter file reference
- [Runtime Arguments](runtime-arguments.md) - Command line options
- [Gridding](gridding.md) - Grid types and usage
- [MPI](mpi.md) - Multi-node execution

For issues or contributions, visit the [GitHub repository](https://github.com/flaresimulations/gridder).
