# Parameter File Reference

The gridder is configured using a YAML parameter file. This page documents all available parameters.

## Complete Example

```yaml
Kernels:
  nkernels: 5
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0
  kernel_radius_4: 4.0
  kernel_radius_5: 8.0

Grid:
  type: uniform
  cdim: 100

Tree:
  max_leaf_count: 200

Input:
  filepath: /path/to/snapshot.hdf5
  placeholder: "0000"

Output:
  filepath: ./output/
  basename: gridded_output.hdf5
  write_masses: 0

Simulation:
  gap_fraction: 0.1
```

## Parameter Sections

### Kernels

Defines the spherical top hat kernels used for gridding.

#### `nkernels`
- **Type**: Integer
- **Required**: Yes
- **Description**: Number of kernel radii to use
- **Range**: 1-10
- **Example**: `nkernels: 5`

#### `kernel_radius_N`
- **Type**: Float
- **Required**: Yes (for N = 1 to nkernels)
- **Description**: Radius of the Nth kernel in simulation units (typically Mpc/h)
- **Units**: Comoving distance (same as simulation)
- **Example**:
  ```yaml
  kernel_radius_1: 0.5  # 500 kpc/h
  kernel_radius_2: 1.0  # 1 Mpc/h
  kernel_radius_3: 2.0  # 2 Mpc/h
  ```

**Notes**:
- Kernels are applied simultaneously in a single pass for efficiency
- Smaller radii give higher resolution but noisier results
- Larger radii give smoother results but lower resolution
- Recommended range: 0.1 - 10.0 Mpc/h for cosmological simulations

---

### Grid

Defines the grid point distribution.

#### `type`
- **Type**: String
- **Required**: Yes
- **Options**: `uniform`, `random`, `file`
- **Description**: Grid point distribution type
- **Example**: `type: uniform`

**Grid Type Details**:

##### `uniform`
Regularly-spaced cubic grid.

**Required parameters**:
- `cdim`: Grid points per dimension (creates cdim³ total points)

**Example**:
```yaml
Grid:
  type: uniform
  cdim: 100  # Creates 100³ = 1,000,000 grid points
```

##### `random`
Randomly-distributed points.

**Required parameters**:
- `n_grid_points`: Total number of grid points

**Example**:
```yaml
Grid:
  type: random
  n_grid_points: 1000000
```

##### `file`
Load grid points from a text file.

**Required parameters**:
- `grid_file`: Path to grid point file

**File format**: ASCII text, one point per line
```
x1 y1 z1
x2 y2 z2
x3 y3 z3
```

**Example**:
```yaml
Grid:
  type: file
  grid_file: /path/to/custom_grid.txt
```

#### `cdim`
- **Type**: Integer
- **Required**: Yes (for `uniform` type only)
- **Description**: Number of grid points per dimension
- **Total points**: cdim³
- **Typical values**: 50-200
- **Example**: `cdim: 100` creates 1,000,000 grid points

#### `n_grid_points`
- **Type**: Integer
- **Required**: Yes (for `random` type only)
- **Description**: Total number of randomly-distributed grid points
- **Typical values**: 100,000 - 10,000,000
- **Example**: `n_grid_points: 1000000`

#### `grid_file`
- **Type**: String (file path)
- **Required**: Yes (for `file` type only)
- **Description**: Path to ASCII file containing grid point coordinates
- **Example**: `grid_file: ./custom_grid.txt`

---

### Tree

Controls the octree spatial partitioning.

#### `max_leaf_count`
- **Type**: Integer
- **Required**: Yes
- **Description**: Maximum number of particles in a leaf cell before subdivision
- **Range**: 50-500
- **Default recommendation**: 200
- **Example**: `max_leaf_count: 200`

**Performance implications**:
- **Lower values** (50-100):
  - Deeper tree (more memory)
  - Faster particle searches
  - Better for large particle counts
- **Higher values** (300-500):
  - Shallower tree (less memory)
  - Slower particle searches
  - Better for limited memory

---

### Input

Specifies the input simulation snapshot.

#### `filepath`
- **Type**: String (file path)
- **Required**: Yes
- **Description**: Path to HDF5 snapshot file
- **Format**: SWIFT-compatible HDF5
- **Example**: `filepath: /data/snapshots/snapshot_0042.hdf5`

**Placeholder substitution**:

If you provide a snapshot number as a command-line argument, the `placeholder` string is replaced:

```yaml
Input:
  filepath: /data/snapshots/snapshot_0000.hdf5
  placeholder: "0000"
```

```bash
# Processes snapshot_0042.hdf5
./build/parent_gridder params.yml 8 42
```

#### `placeholder`
- **Type**: String
- **Required**: No
- **Description**: String in `filepath` to replace with snapshot number
- **Default**: `"0000"`
- **Example**: `placeholder: "XXXX"`

---

### Output

Configures output files.

#### `filepath`
- **Type**: String (directory path)
- **Required**: Yes
- **Description**: Output directory for gridded data
- **Example**: `filepath: ./output/`

**Notes**:
- Directory will be created if it doesn't exist
- In MPI mode, each rank writes to this directory

#### `basename`
- **Type**: String (filename)
- **Required**: Yes
- **Description**: Base filename for output HDF5 files
- **Example**: `basename: gridded_output.hdf5`

**Output filenames**:
- **Serial mode**: `{filepath}/{basename}`
- **MPI mode**: `{filepath}/{basename_without_ext}_rank_{N}.hdf5`

**Examples**:
```yaml
# Serial: output/gridded_output.hdf5
# MPI:    output/gridded_output_rank_0.hdf5
#         output/gridded_output_rank_1.hdf5
```

#### `write_masses`
- **Type**: Integer (boolean)
- **Required**: No
- **Default**: 0
- **Description**: Whether to write total mass in each kernel
- **Values**:
  - `0`: Write only overdensities
  - `1`: Write both overdensities and total masses
- **Example**: `write_masses: 1`

**Impact**:
- Doubles output file size when enabled
- Useful for mass-weighted analyses

---

### Simulation

Simulation-specific parameters.

#### `gap_fraction`
- **Type**: Float
- **Required**: No
- **Default**: 0.0
- **Description**: Fractional gap between top-level cells
- **Range**: 0.0 - 0.5
- **Example**: `gap_fraction: 0.1`

**Purpose**:
- Adds spacing between top-level cells for debugging
- Typically set to 0.0 for production runs
- Values > 0 reduce overlap between adjacent cells

---

## Validation

The parameter parser performs validation:

```cpp
// Missing required parameter
ERROR: Required parameter 'Kernels:nkernels' not found

// Wrong type
ERROR: Parameter 'Grid:cdim' must be an integer

// Invalid kernel count
ERROR: nkernels=15 exceeds maximum of 10

// Missing kernel radius
ERROR: Kernel radius 3 specified but nkernels=2
```

## Example Configurations

### High-Resolution Local Analysis

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.1  # 100 kpc/h
  kernel_radius_2: 0.5  # 500 kpc/h
  kernel_radius_3: 1.0  # 1 Mpc/h

Grid:
  type: uniform
  cdim: 200  # 8M grid points

Tree:
  max_leaf_count: 100  # Deep tree for precision
```

### Large-Volume Survey

```yaml
Kernels:
  nkernels: 5
  kernel_radius_1: 1.0
  kernel_radius_2: 2.0
  kernel_radius_3: 4.0
  kernel_radius_4: 8.0
  kernel_radius_5: 16.0

Grid:
  type: uniform
  cdim: 100  # 1M grid points

Tree:
  max_leaf_count: 300  # Shallow tree for memory
```

### Custom Grid for Galaxy Positions

```yaml
Kernels:
  nkernels: 2
  kernel_radius_1: 0.5
  kernel_radius_2: 2.0

Grid:
  type: file
  grid_file: galaxy_positions.txt

Tree:
  max_leaf_count: 200
```

## See Also

- [Quick Start](quickstart.md) - Getting started tutorial
- [Configuration](configuration.md) - Environment configuration guide
- [Installation](installation.md) - Installation instructions
