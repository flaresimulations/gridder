# FLARES-2 Gridder

The FLARES-2 Gridder computes matter overdensities at uniform, random, or
file-provided grid points using spherical top-hat kernels. The C++20 code uses
an octree for particle searches, OpenMP within a process, and optional MPI
domain decomposition.

## Build

Requirements are CMake 3.12+, a C++20 compiler, OpenMP, and HDF5. MPI is
optional. Serial HDF5 is used in both builds.

```bash
# Serial
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# MPI
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi
```

## Configure

The parameter reader accepts simple indentation-based YAML-style mappings.
See [`example_params.yml`](example_params.yml) and the
[parameter reference](docs/parameters.md).

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
  filepath: snapshot_0000.hdf5
  placeholder: "0000"
  part_gap_fill_fraction: 0.01

Output:
  filepath: output/
  basename: grid_0000.hdf5
  write_masses: 0
```

The cosmology values are required and determine the mean comoving density used
to normalize overdensities.

## Run

```bash
# Eight OpenMP threads, snapshot number defaults to zero
./build/parent_gridder example_params.yml 8

# Snapshot 42
./build/parent_gridder example_params.yml 8 42

# Four MPI ranks, two OpenMP threads per rank
mpirun -n 4 ./build_mpi/parent_gridder example_params.yml 2 42
```

Use `./build/parent_gridder --help` for runtime arguments. The positional
thread argument controls OpenMP in both serial and MPI builds.

## Input Format

The input HDF5 snapshot must contain:

- `Header` attributes `Redshift`, `NumPart_Total`, and `BoxSize`
- `Cells/Meta-data` attributes `dimension` and `size`
- `Cells/Counts/PartType1`
- `Cells/OffsetsInFile/PartType1`
- `PartType1/Coordinates`
- `PartType1/Masses`

Particle IDs, velocities, input units, and an input cosmology group are not
read. The [conversion guide](docs/conversion.md) describes how to create the
required cell index from compatible coordinate and mass datasets.

## Tests

```bash
pytest tests -v
./tests/run_tests.sh --all
```

The shell runner is focused rather than exhaustive. See
[`tests/README.md`](tests/README.md) for serial, MPI, and comparison-suite
commands.

Generate test input manually with:

```bash
python3 tests/make_test_snap.py --help
```

## Documentation

Build the MkDocs site with:

```bash
./build_docs.sh
```

Start with the [quickstart](docs/quickstart.md), then see the
[grid types](docs/gridding.md), [performance and memory model](docs/performance.md),
[MPI guide](docs/mpi.md), and [conversion guide](docs/conversion.md).
