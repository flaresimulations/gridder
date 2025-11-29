```
 ██████╗  █████╗ ██████╗ ███████╗███╗   ██╗████████╗
 ██╔══██╗██╔══██╗██╔══██╗██╔════╝████╗  ██║╚══██╔══╝
 ██████╔╝███████║██████╔╝█████╗  ██╔██╗ ██║   ██║
 ██╔═══╝ ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║   ██║        ┌─────┬─────┬─────┐
 ██║     ██║  ██║██║  ██║███████╗██║ ╚████║   ██║        │  •  │  •  │  •  │
 ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝        ├─────┼─────┼─────┤
                                                         │  •  │  •  │  •  │
  ██████╗ ██████╗ ██╗██████╗ ██████╗ ███████╗██████╗     ├─────┼─────┼─────┤
 ██╔════╝ ██╔══██╗██║██╔══██╗██╔══██╗██╔════╝██╔══██╗    │  •  │  •  │  •  │
 ██║  ███╗██████╔╝██║██║  ██║██║  ██║█████╗  ██████╔╝    └─────┴─────┴─────┘
 ██║   ██║██╔══██╗██║██║  ██║██║  ██║██╔══╝  ██╔══██╗
 ╚██████╔╝██║  ██║██║██████╔╝██████╔╝███████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
```

# Gridding simulations

Apply spherical top hat kernels to the matter distribution of a simulation, creating HDF5 files of overdensities.

This C++ application processes cosmological simulation snapshots to compute overdensities at grid points using multiple spherical top hat kernels. It reads HDF5 simulation files and outputs gridded overdensity data, making it useful for analysing large-scale structure in cosmological simulations.

## Features

- **Multi-kernel analysis**: Apply multiple spherical top hat kernels with different radii simultaneously
- **Efficient spatial partitioning**: Octree-based cell structure for fast neighbour searches
- **HDF5 I/O**: Native support for reading and writing HDF5 simulation data
- **Parallel processing**: Optional MPI support for large simulations
- **OpenMP threading**: Multi-threaded computation within nodes
- **Flexible gridding**: Support for custom grid resolutions and kernel configurations

## Dependencies

- **CMake** 3.12 or higher
- **C++20** compatible compiler
- **HDF5** library (parallel HDF5 required for MPI builds)
- **OpenMP** (typically included with modern compilers)
- **MPI** (optional, for parallel processing)

## Compilation

### Basic Serial Build

```bash
cmake -B build
cmake --build build
```

### Parallel Build with MPI

```bash
cmake -B build -DENABLE_MPI=ON
cmake --build build
```

### Build Configuration Options

The build system supports several configuration options:

- **`ENABLE_MPI`**: Enable MPI parallelisation (default: OFF)
- **`CMAKE_BUILD_TYPE`**: Build type (Release, Debug, RelWithDebInfo, MinSizeRel)

**Build types:**

- `Release` (default): Full optimisation with `-O3 -march=native`
- `Debug`: Debug symbols and checks with `-g -O0`
- `RelWithDebInfo`: Optimised with debug symbols `-O2 -g`
- `MinSizeRel`: Size optimisation with `-Os`

### Example builds:

```bash
# Debug build with MPI
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_MPI=ON
cmake --build build

# Release build (serial)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Single command builds
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_MPI=ON; cmake --build build
```

## Gridding Parameters

The gridder uses YAML configuration files to specify simulation parameters. Here's the parameter file structure:

### Parameter File Format (`example_params.yml`)

```yaml
Kernels:
  nkernels: 5 # Number of different kernel radii
  kernel_radius_1: 0.5 # First kernel radius (Mpc/h)
  kernel_radius_2: 1.0 # Second kernel radius
  kernel_radius_3: 2.0 # Third kernel radius
  kernel_radius_4: 3.0 # Fourth kernel radius
  kernel_radius_5: 5.0 # Fifth kernel radius

Grid:
  type: uniform # Grid point generation type: uniform, random, or file
  cdim: 100 # Number of grid points per dimension (for uniform type, creates 100³ grid)
  n_grid_points: 1000000 # Total number of grid points (for random type)
  grid_file: grid_points.txt # File containing grid points (for file type)

Tree:
  max_leaf_count: 200 # Maximum particles per leaf cell in octree

Input:
  filepath: uniform_test_0000.hdf5 # Input simulation snapshot path
  placeholder: "0000" # Placeholder for numbered snapshots

Output:
  filepath: data/ # Output directory path
  basename: test_grid_0000.hdf5 # Output filename pattern
  write_masses: 0 # Whether to write out mass in a kernel as well as overdensity
```

### Parameter Descriptions

**Kernels section:**

- `nkernels`: Number of different kernel radii to use (1-N)
- `kernel_radius_X`: Radius of kernel X in simulation units (typically Mpc/h)

**Grid section:**

- `type`: Grid point generation method:
  - `uniform`: Regularly spaced grid points (default)
  - `random`: Randomly distributed grid points
  - `file`: Load grid points from a file
- `cdim`: Grid resolution per dimension for uniform grids (total grid points = cdim³)
- `n_grid_points`: Total number of random grid points to generate (for random type)
- `grid_file`: Path to file containing grid point coordinates (for file type)

**Tree section:**

- `max_leaf_count`: Controls octree subdivision (lower values = deeper trees, higher memory usage)

**Input section:**

- `filepath`: Path to input HDF5 snapshot file
- `placeholder`: String pattern for multi-snapshot processing

**Output section:**

- `filepath`: Output directory for gridded data
- `basename`: Output filename template

## Running the gridder

### Basic Usage

```bash
./build/parent_gridder <parameter_file> <nthreads> [snapshot_number]
```

**Arguments:**

- `parameter_file`: YAML configuration file path
- `nthreads`: Number of OpenMP threads to use
- `snapshot_number`: Optional snapshot number (replaces placeholder in filenames)

### Examples

**Process single snapshot with 8 threads, processing exactly the snapshot specified in the parameter file:**

```bash
./build/parent_gridder example_params.yml 8
```

**Process specific snapshot (e.g., snapshot 42):**

```bash
./build/parent_gridder params.yml 16 42
```

**With MPI (4 processes, 8 threads each), processing exactly the snapshot specified in the parameter file:**

```bash
mpirun -n 4 ./build/parent_gridder example_params.yml 8
```

### Input File Requirements

The input HDF5 files should contain:

- `PartType1/Coordinates`: Particle positions
- `PartType1/ParticleIDs`: Particle IDs
- `Header`: Simulation metadata (box size, particle counts, etc.)
- `Cells`: Cell structure information (if available)
- `Units`: Unit system information
- `Cosmology`: Cosmological parameters

Currently, this is built on the back of [SWIFT](https://swift.strw.leidenuniv.nl/docs/index.html) outputs, but can be adapted for other formats with minor changes.

### Output Files

The gridder produces HDF5 files containing:

- Overdensity values at each grid point for all kernel radii
- Grid geometry and metadata
- Kernel radius information
- Processing metadata and timestamps

## Converting Arbitrary Snapshots

Use the conversion tool to process snapshots from any simulation code:

```bash
python tools/convert_to_gridder_format.py input.hdf5 output.hdf5 \
  --coordinates-key PartType1/Coordinates \
  --masses-key PartType1/Masses \
  --copy-header
```

**Key features:**
- Handles arbitrary HDF5 dataset names
- Creates required cell structure automatically
- Supports both cubic and non-cubic simulation boxes
- MPI parallelization for large files
- Configurable cell grid resolution (default: 16×16×16)

**See [Conversion Guide](docs/conversion.md) for detailed documentation.**

## Generating Test Data

Use the included Python script to create test snapshots:

```bash
python make_test_snap.py \
  --output uniform_test_0000.hdf5 \
  --cdim 100 \
  --grid_dim 50 \
  --boxsize 10.0 \
  --doner /path/to/reference/snapshot.hdf5
```

**Parameters:**

- `--output`: Output filename for test snapshot
- `--cdim`: Number of cells per dimension
- `--grid_dim`: Number of particles per dimension
- `--boxsize`: Size of simulation box
- `--doner`: Reference snapshot for units and cosmology
