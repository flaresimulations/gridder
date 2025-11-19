# Installation

## Requirements

### Essential Dependencies

- **C++20 Compiler**: GCC 10+, Clang 12+, or equivalent
- **CMake**: Version 3.12 or higher
- **HDF5**: Serial HDF5 library (C and HL components)
- **OpenMP**: Threading library (usually bundled with compiler)

### Optional Dependencies

- **MPI**: For multi-node parallelization (OpenMPI, MPICH, or Intel MPI)

## Platform-Specific Installation

### macOS (Homebrew)

```bash
# Install dependencies
brew install cmake hdf5 libomp

# Optional: Install MPI
brew install open-mpi
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake libhdf5-dev

# OpenMP usually comes with GCC

# Optional: Install MPI
sudo apt-get install libopenmpi-dev openmpi-bin
```

### Linux (CentOS/RHEL)

```bash
# Install dependencies
sudo yum install gcc-c++ cmake hdf5-devel

# Optional: Install MPI
sudo yum install openmpi-devel
module load mpi/openmpi-x86_64
```

## Building from Source

### Single-Node Build (OpenMP only)

```bash
# Clone repository
git clone https://github.com/flaresimulations/gridder.git
cd gridder

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8

# Verify build
./build/parent_gridder --help
```

### Multi-Node Build (MPI + OpenMP)

```bash
# Configure with MPI support
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi -j 8

# Verify build
mpirun -n 2 ./build_mpi/parent_gridder --help
```

## Build Types

| Build Type | Flags | Description |
|------------|-------|-------------|
| `Release` (default) | `-O3 -march=native` | Maximum performance |
| `Debug` | `-g -O0` | Debugging symbols, assertions enabled |
| `RelWithDebInfo` | `-O2 -g` | Optimized with debug symbols |
| `MinSizeRel` | `-Os` | Optimized for binary size |

Example with debug build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Verification

### Check OpenMP Support

```bash
# macOS
otool -L build/parent_gridder | grep omp

# Linux
ldd build/parent_gridder | grep omp
```

Expected output: Should show `libomp` or `libgomp` linked.

### Check MPI Support

```bash
# macOS
otool -L build_mpi/parent_gridder | grep -E 'mpi|omp'

# Linux
ldd build_mpi/parent_gridder | grep -E 'mpi|omp'
```

Expected output: Should show both MPI and OpenMP libraries.

### Run Test Suite

```bash
# Test single-node build
python tests/test_suite.py --mode serial

# Test multi-node build
python tests/test_suite.py --mode mpi --ranks 2
```

All tests should pass (12/12 for serial, 17/17 for MPI).

## Troubleshooting

### HDF5 Not Found

If CMake can't find HDF5:

```bash
# Specify HDF5 location explicitly
cmake -B build -DHDF5_ROOT=/path/to/hdf5
```

### OpenMP Not Found

If OpenMP detection fails:

```bash
# macOS: Specify libomp location
cmake -B build -DOpenMP_ROOT=/opt/homebrew/opt/libomp

# Linux: Ensure compiler supports OpenMP
export CC=gcc-10
export CXX=g++-10
cmake -B build
```

### MPI Compiler Issues

If MPI compilers aren't detected:

```bash
# Specify MPI compilers explicitly
cmake -B build_mpi -DENABLE_MPI=ON \
  -DMPI_C_COMPILER=mpicc \
  -DMPI_CXX_COMPILER=mpicxx
```

## Next Steps

- [Quick Start Guide](quickstart.md): Run your first simulation
- [Configuration](configuration.md): Set up parameter files
