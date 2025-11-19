# Parameter Reference

Complete reference for all parameter file options.

## Overview

Parameter files use YAML syntax and define:

- **Kernels**: Smoothing radii for overdensity calculations
- **Grid**: Grid point generation method and resolution
- **Tree**: Octree spatial indexing configuration
- **Input**: Simulation snapshot file path
- **Output**: Output file location and options

## Complete Example

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0

Grid:
  type: uniform
  cdim: 100

Tree:
  max_leaf_count: 200

Input:
  filepath: /path/to/snapshot_0042.hdf5
  placeholder: "0042"

Output:
  filepath: ./output/
  basename: gridded_data.hdf5
  write_masses: 1
```

---

## Kernels

Defines the spherical top hat kernel radii for overdensity calculations.

### `nkernels`

!!! info "Required Parameter"

**Type:** Integer
**Range:** 1 to 100 (practical limit)
**Description:** Number of kernel radii to use

Each kernel radius produces a separate overdensity field. Multiple radii enable multi-scale analysis in a single run.

**Example:**
```yaml
Kernels:
  nkernels: 5  # Will use 5 different kernel radii
```

**Performance Note:** Each kernel adds ~linear overhead. 10 kernels ≈ 10× slower than 1 kernel.

### `kernel_radius_N`

!!! info "Required Parameters (N=1 to nkernels)"

**Type:** Float
**Units:** Comoving Mpc/h (simulation units)
**Range:** > 0.0
**Description:** Radius of the N-th spherical top hat kernel

Kernel radii must be numbered consecutively starting from 1. The gridder counts particles within a sphere of this radius around each grid point.

**Example:**
```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5   # Small scale (0.5 Mpc/h)
  kernel_radius_2: 1.0   # Medium scale (1.0 Mpc/h)
  kernel_radius_3: 2.0   # Large scale (2.0 Mpc/h)
```

**Physics Notes:**

- Typical values: 0.1 - 10.0 Mpc/h
- Smaller radii: Higher resolution, noisier
- Larger radii: Smoother fields, lower resolution
- Choose based on science goals (e.g., halo masses, large-scale structure)

**Performance Impact:**

- Larger radii = more particles per grid point = slower
- Octree optimization helps, but 10 Mpc/h kernels will be much slower than 0.5 Mpc/h

---

## Grid

Defines how grid points are generated or loaded.

### `type`

!!! info "Required Parameter"

**Type:** String
**Options:** `uniform`, `random`, `file`
**Description:** Method for generating grid points

See [Gridding](gridding.md) for detailed descriptions of each type.

**Example:**
```yaml
Grid:
  type: uniform  # Regular cubic grid
```

### `cdim` {#grid-cdim}

!!! info "Required when `type: uniform`"

**Type:** Integer
**Range:** 1 to ~1000 (limited by memory)
**Description:** Grid points per dimension (creates `cdim³` total points)

**Example:**
```yaml
Grid:
  type: uniform
  cdim: 100  # Creates 100³ = 1,000,000 grid points
```

**Memory estimate:** ~80 bytes per grid point (depends on nkernels)

```
cdim=50:   125,000 points ≈ 10 MB
cdim=100:  1,000,000 points ≈ 80 MB
cdim=200:  8,000,000 points ≈ 640 MB
cdim=500:  125,000,000 points ≈ 10 GB
```

### `n_grid_points`

!!! info "Required when `type: random`"

**Type:** Integer
**Range:** 1 to ~100,000,000 (limited by memory)
**Description:** Total number of random grid points to generate

**Example:**
```yaml
Grid:
  type: random
  n_grid_points: 1000000  # 1 million random points
```

Random points are uniformly distributed within the simulation box using a Mersenne Twister RNG.

### `grid_file`

!!! info "Required when `type: file`"

**Type:** String (file path)
**Description:** Path to text file containing grid point coordinates

**Example:**
```yaml
Grid:
  type: file
  grid_file: /path/to/grid_points_0042.txt
```

**File Format:**

```
# Comments start with #
# Format: x y z (one point per line)
10.5 20.3 15.7
25.0 30.0 35.0
# Whitespace and empty lines are ignored
12.1 18.9 22.3
```

**Coordinate Units:** Same as simulation (typically comoving Mpc/h)

**Placeholder Support:** Grid file paths support snapshot placeholders (see [Input/placeholder](#input-placeholder))

**Example with placeholder:**
```yaml
Input:
  placeholder: "0000"

Grid:
  type: file
  grid_file: /data/grids/grid_points_0000.txt
  # When snapshot 42 is specified: grid_points_0042.txt
```

---

## Tree

Configures the octree spatial indexing structure.

### `max_leaf_count`

!!! info "Required Parameter"

**Type:** Integer
**Range:** 1 to 10000 (practical: 50-500)
**Description:** Maximum particles allowed in a leaf cell before subdivision

The octree recursively splits cells containing more than `max_leaf_count` particles until all cells meet this threshold or maximum depth is reached.

**Example:**
```yaml
Tree:
  max_leaf_count: 200
```

**Performance Trade-offs:**

| Value | Tree Depth | Search Speed | Memory Usage | Build Time |
|-------|------------|--------------|--------------|------------|
| 50 | Deeper | Faster | Higher | Slower |
| 200 | Medium | Medium | Medium | Medium |
| 500 | Shallower | Slower | Lower | Faster |

**Recommendations:**

- **Default:** 200 (good balance)
- **Memory constrained:** 300-500
- **Speed critical:** 100-150
- **Very large simulations:** 200-300

**Technical Details:**

Lower values create deeper trees with more cells but faster particle searches. Higher values create shallower trees with fewer cells but slower searches within each cell.

Maximum tree depth is logged during execution:
```
[INFO] Maximum depth in the tree: 12
```

---

## Input

Defines the simulation snapshot file to process.

### `filepath`

!!! info "Required Parameter"

**Type:** String (file path)
**Description:** Path to HDF5 simulation snapshot

**Example:**
```yaml
Input:
  filepath: /scratch/simulations/FLARES/snapshot_0042.hdf5
```

**Supported Formats:**

- SWIFT HDF5 snapshots (primary target)
- Must contain `PartType1` group with `Coordinates` and `Masses` datasets
- Must contain `Header` with `BoxSize`, `Redshift` attributes
- Must contain `Units` and `Cosmology` groups

**Placeholder Replacement:**

If snapshot number is provided as command line argument, placeholders in the path are replaced:

```yaml
Input:
  filepath: /data/snapshots/snap_0000.hdf5
  placeholder: "0000"
```

Command: `./parent_gridder params.yml 8 42`
Actual path: `/data/snapshots/snap_0042.hdf5`

### `placeholder` {#input-placeholder}

!!! info "Optional Parameter"

**Type:** String
**Default:** `"0000"`
**Description:** String pattern in file paths to replace with snapshot number

**Example:**
```yaml
Input:
  filepath: /data/snapshot_XXXX.hdf5
  placeholder: "XXXX"
```

Command: `./parent_gridder params.yml 8 42`
Result: `/data/snapshot_0042.hdf5`

**Padding:**

The snapshot number is zero-padded to match the placeholder length:

```yaml
placeholder: "000"   # 3 digits: snap 5 → snap_005.hdf5
placeholder: "0000"  # 4 digits: snap 5 → snap_0005.hdf5
```

**Multiple Occurrences:**

All instances of the placeholder in the path are replaced:

```yaml
filepath: /data/0000/snapshot_0000.hdf5
placeholder: "0000"
# snap 42 → /data/0042/snapshot_0042.hdf5
```

**Applies To:**

- `Input/filepath`
- `Output/basename`
- `Grid/grid_file` (when `type: file`)

---

## Output

Configures output file location and content.

### `filepath`

!!! info "Required Parameter"

**Type:** String (directory path)
**Description:** Directory where output files will be written

**Example:**
```yaml
Output:
  filepath: ./output/
```

**Notes:**

- Directory must exist (not created automatically)
- Must have write permissions
- For MPI builds, per-rank files are written here (e.g., `output_rank0.hdf5`)

### `basename`

!!! info "Required Parameter"

**Type:** String (filename)
**Description:** Base name for output HDF5 file(s)

**Example:**
```yaml
Output:
  filepath: ./results/
  basename: gridded_snapshot.hdf5
```

Output: `./results/gridded_snapshot.hdf5`

**MPI Mode:**

Per-rank files + virtual file:
```
./results/gridded_snapshot_rank0.hdf5
./results/gridded_snapshot_rank1.hdf5
./results/gridded_snapshot_rank2.hdf5
./results/gridded_snapshot.hdf5  # Virtual file combining all ranks
```

**Placeholder Support:**

```yaml
Output:
  basename: gridded_snap_0000.hdf5

Input:
  placeholder: "0000"
```

Command: `./parent_gridder params.yml 8 42`
Output: `gridded_snap_0042.hdf5`

### `write_masses`

!!! info "Optional Parameter"

**Type:** Integer (0 or 1)
**Default:** `0`
**Description:** Whether to write total mass within kernels

**Values:**

- `0`: Write only overdensities (default, saves space)
- `1`: Write both overdensities and masses

**Example:**
```yaml
Output:
  write_masses: 1  # Include mass datasets
```

**Output Datasets:**

When `write_masses: 0`:
```
/Grids/Kernel_0/GridPointOverDensities
/Grids/Kernel_1/GridPointOverDensities
...
```

When `write_masses: 1`:
```
/Grids/Kernel_0/GridPointOverDensities
/Grids/Kernel_0/GridPointMasses
/Grids/Kernel_1/GridPointOverDensities
/Grids/Kernel_1/GridPointMasses
...
```

**Mass Units:** $10^{10}$ M$_\odot$

**File Size Impact:** Approximately doubles file size when enabled.

---

## Complete Parameter Examples

### Example 1: Quick Uniform Grid

Minimal setup for testing:

```yaml
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: uniform
  cdim: 50  # 125k points

Tree:
  max_leaf_count: 200

Input:
  filepath: snapshot.hdf5

Output:
  filepath: ./
  basename: test_grid.hdf5
  write_masses: 0
```

### Example 2: Production Multi-Scale Analysis

High-resolution with multiple scales:

```yaml
Kernels:
  nkernels: 5
  kernel_radius_1: 0.25
  kernel_radius_2: 0.5
  kernel_radius_3: 1.0
  kernel_radius_4: 2.0
  kernel_radius_5: 4.0

Grid:
  type: uniform
  cdim: 200  # 8 million points

Tree:
  max_leaf_count: 150  # Deeper tree for speed

Input:
  filepath: /data/FLARES/snapshots/snap_0000.hdf5
  placeholder: "0000"

Output:
  filepath: /scratch/output/
  basename: FLARES_gridded_0000.hdf5
  write_masses: 1
```

### Example 3: Random Sampling for Statistics

Monte Carlo approach:

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0

Grid:
  type: random
  n_grid_points: 5000000  # 5 million random points

Tree:
  max_leaf_count: 200

Input:
  filepath: snapshot_0050.hdf5

Output:
  filepath: ./random_grids/
  basename: random_grid_0050.hdf5
  write_masses: 0
```

### Example 4: Custom Grid Points

Target specific regions:

```yaml
Kernels:
  nkernels: 2
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0

Grid:
  type: file
  grid_file: /path/to/halo_centers_0000.txt

Tree:
  max_leaf_count: 200

Input:
  filepath: /data/snapshot_0000.hdf5
  placeholder: "0000"

Output:
  filepath: /results/halo_grids/
  basename: halo_grid_0000.hdf5
  write_masses: 1
```

---

## Validation

The gridder validates all parameters at startup and reports errors:

**Missing required parameter:**
```
ERROR: A required parameter was not set in the parameter file (Kernels/nkernels)
```

**Invalid value:**
```
ERROR: Invalid grid type specified: uniformm
(Did you mean 'uniform'?)
```

**File not found:**
```
ERROR: Failed to open grid points file: /path/to/missing.txt
```

See [Troubleshooting](quickstart.md#troubleshooting) for common issues.

## See Also

- **[Gridding](gridding.md)** - Detailed grid type descriptions
- **[Runtime Arguments](runtime-arguments.md)** - Command line options
- **[Quickstart](quickstart.md)** - Get started quickly
