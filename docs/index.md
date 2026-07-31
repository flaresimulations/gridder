# FLARES-2 Gridder

The FLARES-2 Gridder computes matter overdensities at user-defined grid points
with one or more spherical top-hat kernels. It reads cell-indexed HDF5
snapshots and writes gridded HDF5 output.

## Features

- Uniform, random, and file-based grid points
- Multiple kernel radii in one run
- Octree particle searches
- OpenMP threading and optional MPI domain decomposition
- Chunked reads that avoid loading particles far from requested grid points
- Serial snapshot conversion for compatible HDF5 inputs

## Quick Example

```bash
# Eight OpenMP threads
./build/parent_gridder params.yml 8

# Four MPI ranks, two OpenMP threads per rank
mpirun -n 4 ./build_mpi/parent_gridder params.yml 2
```

The second positional argument always sets the OpenMP thread count, including
in MPI builds.

## Documentation

- [Quickstart](quickstart.md)
- [Installation](installation.md)
- [Parameter reference](parameters.md)
- [Runtime arguments](runtime-arguments.md)
- [Grid types](gridding.md)
- [MPI execution](mpi.md)
- [Snapshot conversion](conversion.md)

Report problems through the [GitHub repository](https://github.com/flaresimulations/gridder).
