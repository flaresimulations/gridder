# Installation

Complete installation guide for the FLARES-2 Gridder with various build configurations.

## System Requirements

### Required Dependencies

| Dependency | Minimum Version | Purpose |
|-----------|----------------|---------|
| **CMake** | 3.12+ | Build system |
| **C++ Compiler** | C++20 support | Compilation (GCC 10+, Clang 11+, AppleClang 13+) |
| **HDF5** | 1.10+ | Snapshot I/O |
| **OpenMP** | 4.5+ | Multi-threading |

### Optional Dependencies

| Dependency | Purpose | When to Use |
|-----------|---------|-------------|
| **MPI** | Multi-node parallelization | Simulations >10GB or >16 cores needed |
| **Python 3** | Test suite, utilities | Development and testing |
| **h5py** | Test data generation | Creating test snapshots |

## Quick Install

=== "Ubuntu/Debian"

    ```bash
    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y \
        cmake \
        g++ \
        libhdf5-dev \
        libomp-dev

    # Clone repository
    git clone https://github.com/flaresimulations/gridder.git
    cd gridder

    # Build (single-node, OpenMP only)
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ```

=== "macOS (Homebrew)"

    ```bash
    # Install dependencies
    brew install cmake hdf5 libomp

    # Clone repository
    git clone https://github.com/flaresimulations/gridder.git
    cd gridder

    # Build (single-node, OpenMP only)
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ```

=== "HPC Module System"

    ```bash
    # Load modules (adjust names for your system)
    module load cmake/3.20
    module load gcc/11.2
    module load hdf5/1.12
    module load openmpi/4.1  # Optional, for MPI build

    # Clone repository
    git clone https://github.com/flaresimulations/gridder.git
    cd gridder

    # Build (see build modes below)
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ```

## Build Modes

The gridder supports different build configurations depending on your needs.

### Single-Node Build (OpenMP Threading)

Best for: Desktop use, small simulations (<10GB), up to ~16 cores

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Features:**
- Multi-threaded with OpenMP
- Shared memory parallelization
- Simple execution (no MPI required)
- Single output HDF5 file

**Executable:** `build/parent_gridder`

**Usage:**
```bash
# Auto-detect number of cores
./build/parent_gridder params.yml 1

# Specify 8 OpenMP threads explicitly
./build/parent_gridder params.yml 8
```

### Multi-Node Build (MPI + OpenMP Hybrid)

Best for: HPC clusters, large simulations (>10GB), hundreds of cores

```bash
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi
```

**Features:**
- Distributed memory parallelization (MPI)
- Per-rank multi-threading (OpenMP)
- Domain decomposition with ghost cells
- Per-rank HDF5 output files + virtual file

**Executable:** `build_mpi/parent_gridder`

**Usage:**
```bash
# 4 MPI ranks × 2 OpenMP threads = 8 cores total
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1

# 16 ranks × 4 threads = 64 cores
export OMP_NUM_THREADS=4
mpirun -n 16 ./build_mpi/parent_gridder params.yml 1
```

### Debug Build

For development and debugging:

```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug
```

**Features:**
- Debug symbols (`-g`)
- No optimization (`-O0`)
- Additional runtime checks (`DEBUGGING_CHECKS` enabled)
- Verbose error messages

**Note:** 10-100× slower than Release build

### Build Types Summary

| Build Type | Optimization | Debug Symbols | Use Case |
|-----------|-------------|---------------|----------|
| `Release` | `-O3 -march=native` | No | Production (default) |
| `Debug` | `-O0` | Yes | Development, debugging |
| `RelWithDebInfo` | `-O2` | Yes | Performance profiling |
| `MinSizeRel` | `-Os` | No | Minimal binary size |

## Advanced Configuration

### Specifying Compiler

```bash
# Use specific compiler
CC=gcc-11 CXX=g++-11 cmake -B build
cmake --build build
```

### Custom HDF5 Location

If HDF5 is not found automatically:

```bash
cmake -B build \
  -DHDF5_ROOT=/path/to/hdf5 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Custom Build Flags

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -mavx2" \  # Custom optimization
  -DCMAKE_INSTALL_PREFIX=/opt/gridder  # Install location
cmake --build build
```

### Install to System

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build  # Installs to CMAKE_INSTALL_PREFIX
```

## Verification

### Test the Build

Quick test with the simple test suite:

```bash
# Single-node build
bash tests/run_simple_test.sh

# MPI build (requires MPI)
cd tests && bash run_mpi_tests.sh
```

### Check Version

```bash
./build/parent_gridder --help
```

Should display:
```
FLARES-2 Parent Gridder
Version: 0.1.0
Git: <commit> (<branch>)

Usage: parent_gridder <parameter_file> <nthreads> [snapshot_number] [verbosity]
...
```

## Troubleshooting

??? failure "CMake can't find HDF5"

    **Symptoms:**
    ```
    Could NOT find HDF5 (missing: HDF5_LIBRARIES HDF5_INCLUDE_DIRS)
    ```

    **Solutions:**

    1. Install HDF5 development headers:
       ```bash
       # Ubuntu/Debian
       sudo apt-get install libhdf5-dev

       # macOS
       brew install hdf5
       ```

    2. Specify HDF5 location explicitly:
       ```bash
       cmake -B build -DHDF5_ROOT=/path/to/hdf5
       ```

    3. On HPC, load HDF5 module:
       ```bash
       module load hdf5
       module show hdf5  # Shows HDF5_ROOT path
       ```

??? failure "OpenMP not found"

    **Symptoms:**
    ```
    Could NOT find OpenMP_CXX
    ```

    **Solutions:**

    1. Install OpenMP:
       ```bash
       # Ubuntu/Debian
       sudo apt-get install libomp-dev

       # macOS
       brew install libomp
       ```

    2. For macOS with AppleClang, ensure libomp is in path:
       ```bash
       export OpenMP_ROOT=$(brew --prefix)/opt/libomp
       cmake -B build -DOpenMP_ROOT=$OpenMP_ROOT
       ```

??? failure "C++20 not supported"

    **Symptoms:**
    ```
    error: This file requires compiler and library support for the ISO C++ 2020 standard.
    ```

    **Solution:** Update your compiler:
    ```bash
    # Ubuntu/Debian
    sudo apt-get install g++-11  # Or g++-12, g++-13
    CC=gcc-11 CXX=g++-11 cmake -B build
    ```

??? failure "MPI build fails"

    **Symptoms:**
    ```
    Could NOT find MPI_CXX
    ```

    **Solutions:**

    1. Install MPI:
       ```bash
       # Ubuntu/Debian
       sudo apt-get install libopenmpi-dev

       # macOS
       brew install open-mpi
       ```

    2. Load MPI module on HPC:
       ```bash
       module load openmpi  # or mpich, intel-mpi, etc.
       ```

    3. Specify MPI compiler explicitly:
       ```bash
       cmake -B build_mpi \
         -DENABLE_MPI=ON \
         -DMPI_CXX_COMPILER=mpicxx
       ```

??? failure "Link errors with HDF5"

    **Symptoms:**
    ```
    undefined reference to `H5Fopen'
    ```

    **Solution:** Ensure HDF5 C and HL (high-level) libraries are linked. On some systems:
    ```bash
    cmake -B build \
      -DHDF5_USE_STATIC_LIBRARIES=OFF \
      -DCMAKE_BUILD_TYPE=Release
    ```

## Next Steps

- **[Quickstart](quickstart.md)** - Get started with a simple example
- **[Parameters](parameters.md)** - Learn about parameter file configuration
- **[Runtime Arguments](runtime-arguments.md)** - Understand command line options
- **[MPI](mpi.md)** - Detailed MPI usage and optimization

## Build Configuration Summary

After successful build, CMake displays a configuration summary:

```
=== Configuration Summary ===
Project: ZoomParentGridder
Version: 0.1.0
Build type: Release
C++ standard: 20
Compiler: GNU@11.2.0
MPI enabled: ON/OFF
HDF5 version: 1.12.1
OpenMP enabled: TRUE
Git revision: abc1234
Git branch: main
===============================
```

Verify this matches your requirements before proceeding.
