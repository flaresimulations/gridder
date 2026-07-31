# Quickstart

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

See [Installation](installation.md) for dependency and MPI build details.

## Create A Parameter File

The parameter reader accepts simple indentation-based YAML-style mappings.
Create `params.yml`:

```yaml
Kernels:
  nkernels: 2
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0

Grid:
  type: uniform
  cdim: 50

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: /path/to/snapshot_0042.hdf5

Output:
  filepath: ./output/
  basename: gridded_snapshot.hdf5
  write_masses: 0
```

The cosmology values are required. They determine the mean comoving matter
density used to normalize overdensities; they are not read from the snapshot.

## Run

```bash
# Eight OpenMP threads
./build/parent_gridder params.yml 8

# Four MPI ranks and two OpenMP threads per rank
mpirun -n 4 ./build_mpi/parent_gridder params.yml 2
```

The output directory is created when it does not exist. A serial run writes
`output/gridded_snapshot.hdf5`. An MPI run also writes rank files and then
combines them into that output file; see [MPI](mpi.md).

## Input Snapshot

The gridder currently reads dark matter from `PartType1`. A compatible HDF5
file contains:

- `Header` attributes `Redshift`, `NumPart_Total`, and `BoxSize`
- `Cells/Meta-data` attributes `dimension` and `size`
- `Cells/Counts/PartType1`
- `Cells/OffsetsInFile/PartType1`
- `PartType1/Coordinates`
- `PartType1/Masses`

Particle IDs, velocities, input `Units`, and an input `Cosmology` group are not
read. Use the [conversion tool](conversion.md) when the required cell index is
absent.

## Next Steps

- [Parameter reference](parameters.md)
- [Runtime arguments](runtime-arguments.md)
- [Grid types](gridding.md)
- [MPI execution](mpi.md)
