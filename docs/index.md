# FLARES-2 Gridder

High-performance cosmological simulation gridding with spherical top hat kernels.

## Overview

The FLARES-2 Gridder is a C++ application designed for gridding cosmological simulations. It applies spherical top hat kernels to matter distributions, reading HDF5 snapshot files from simulations (primarily SWIFT outputs) and computing overdensities at grid points using multiple kernel radii simultaneously.

### Key Features

- **Multiple Kernel Radii**: Compute overdensities for multiple smoothing scales in a single pass
- **Hybrid Parallelization**: OpenMP threading + optional MPI for distributed memory
- **Efficient I/O**: Chunked HDF5 reading with automatic optimization for sparse/dense grids
- **Flexible Grids**: Support for uniform, random, and file-based grid point distributions
- **Octree Spatial Indexing**: Hierarchical cell structure for efficient neighbor searches
- **Production Ready**: Comprehensive test suite with 29 tests covering serial and MPI modes

### Performance Characteristics

| Feature | Single-Node (OpenMP) | Multi-Node (MPI + OpenMP) |
|---------|---------------------|---------------------------|
| Parallelization | Multi-threaded | Distributed + Multi-threaded |
| Memory | Shared | Distributed with ghost cells |
| I/O | Chunked HDF5 | Per-rank HDF5 files |
| Scalability | Up to ~16 cores | Hundreds of cores |

### Quick Example

```bash
# Single-node with 8 OpenMP threads
export OMP_NUM_THREADS=8
./build/parent_gridder params.yml 1

# Multi-node: 4 MPI ranks × 2 OpenMP threads = 8 cores
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation, quick start, and configuration
- **[Parameter Reference](getting-started/parameters.md)**: Detailed parameter file documentation
- **[Performance](performance/openmp.md)**: OpenMP and MPI optimization guides

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Parameter File Reference](getting-started/parameters.md)
- [OpenMP Threading](performance/openmp.md)
- [MPI Parallelization](performance/mpi.md)

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/flaresimulations/gridder).
