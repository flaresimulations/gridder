# Installation

## Requirements

- CMake 3.12 or newer
- A C++20 compiler
- OpenMP
- HDF5 with C and high-level libraries
- MPI for an MPI build
- Python 3, pytest, h5py, and numpy for tests and conversion tools

The MPI build uses ordinary serial HDF5 files; parallel HDF5 is not required.

## Serial Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The executable is `build/parent_gridder`.

## MPI Build

```bash
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi
```

The executable is `build_mpi/parent_gridder`.

## Debug Build

```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug
```

Debug builds enable symbols, warnings, and additional runtime checks.

## CMake Options

| Option | Description |
|--------|-------------|
| `ENABLE_MPI` | Enable MPI communication; default `OFF` |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug`, `RelWithDebInfo`, or `MinSizeRel` |

Use a specific compiler or HDF5 installation through normal CMake variables:

```bash
CC=gcc-13 CXX=g++-13 cmake -B build
cmake -B build -DHDF5_ROOT=/path/to/hdf5
```

The project does not currently define a CMake install target.

## Verify

```bash
./build/parent_gridder --help
pytest tests -v
```

For the custom MPI suite:

```bash
python3 tests/test_suite.py --mode mpi \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2
```

See `tests/README.md` in the repository for the complete test entry points.

## Common Configuration Problems

### HDF5 Not Found

Pass `-DHDF5_ROOT=/path/to/hdf5`, or ensure `h5cc` and the HDF5 C libraries are
available in the environment.

### OpenMP On macOS

Apple Clang does not ship its own OpenMP runtime. Install an OpenMP-capable
compiler or `libomp`, then point CMake at the resulting compiler/runtime.

### MPI Not Found

Load or install an MPI implementation and ensure `mpicxx` is on `PATH` before
configuring with `-DENABLE_MPI=ON`.
