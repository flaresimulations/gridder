# Installation

## Requirements

**Required:**
- CMake 3.12+
- C++20 compiler (GCC 10+, Clang 11+, AppleClang 13+)
- HDF5 1.10+
- OpenMP 4.5+

**Optional:**
- MPI (for multi-node)
- Python 3 + h5py (for tests)

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

### Single-Node (OpenMP)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/parent_gridder params.yml 8
```

Use for: Small simulations (<10GB), up to ~16 cores

### Multi-Node (MPI + OpenMP)

```bash
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi

export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```

Use for: Large simulations (>10GB), HPC clusters

### Debug Build

```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug
```

Enables runtime checks and debug symbols. 10-100× slower than Release.

## Advanced Configuration

```bash
# Custom compiler
CC=gcc-11 CXX=g++-11 cmake -B build

# Custom HDF5 location
cmake -B build -DHDF5_ROOT=/path/to/hdf5

# Custom optimization flags
cmake -B build -DCMAKE_CXX_FLAGS="-O3 -mavx2"
```

## Verification

```bash
# Run tests
bash tests/run_simple_test.sh

# Check version
./build/parent_gridder --help
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

- [Quickstart](quickstart.md) - Basic usage
- [Parameters](parameters.md) - Parameter file format
- [MPI](mpi.md) - Multi-node execution
