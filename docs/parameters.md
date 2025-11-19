# Parameter Reference

Parameter file format (YAML):

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

### `nkernels`

**Type:** Integer (1-100)
**Required**

Number of kernel radii. Each kernel produces a separate overdensity field.

### `kernel_radius_N`

**Type:** Float (> 0.0)
**Units:** Comoving Mpc/h
**Required:** N=1 to nkernels

Radius of N-th spherical top hat kernel. Must be numbered consecutively from 1.

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0
```

Typical range: 0.1 - 10.0 Mpc/h. Larger radii are slower (more particles per grid point).

---

## Grid

### `type`

**Type:** String (`uniform`, `random`, `file`)
**Required**

Grid generation method. See [Gridding](gridding.md) for details.

### `cdim`

**Type:** Integer (1-1000)
**Required when `type: uniform`**

Grid points per dimension. Creates `cdim³` total points.

```yaml
Grid:
  type: uniform
  cdim: 100  # 1,000,000 points, ~80 MB
```

### `n_grid_points`

**Type:** Integer
**Required when `type: random`**

Total random grid points to generate.

### `grid_file`

**Type:** String (file path)
**Required when `type: file`**

Path to text file with coordinates (x y z, one per line). Supports snapshot placeholder replacement.

---

## Tree

### `max_leaf_count`

**Type:** Integer (1-10000, typically 50-500)
**Required**

Maximum particles in a leaf cell before subdivision.

```yaml
Tree:
  max_leaf_count: 200  # Good default
```

Lower values = deeper tree, faster search, more memory. Typical range: 100-300.

---

## Input

### `filepath`

**Type:** String (file path)
**Required**

Path to HDF5 simulation snapshot (SWIFT format). Supports placeholder replacement with snapshot number.

### `placeholder`

**Type:** String
**Optional (default: `"0000"`)**

Pattern in file paths to replace with zero-padded snapshot number.

```yaml
Input:
  filepath: /data/snapshot_0000.hdf5
  placeholder: "0000"
```

Command: `./parent_gridder params.yml 8 42` → `/data/snapshot_0042.hdf5`

Applies to: `Input/filepath`, `Output/basename`, `Grid/grid_file`

---

## Output

### `filepath`

**Type:** String (directory path)
**Required**

Output directory. Must exist and be writable.

### `basename`

**Type:** String (filename)
**Required**

Output HDF5 filename. Supports placeholder replacement. In MPI mode, creates per-rank files plus virtual file.

### `write_masses`

**Type:** Integer (0 or 1)
**Optional (default: 0)**

Write mass within kernels in addition to overdensities. Units: $10^{10}$ M$_\odot$. Doubles file size when enabled.

---

## Examples

Quick test:
```yaml
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0
Grid:
  type: uniform
  cdim: 50
Tree:
  max_leaf_count: 200
Input:
  filepath: snapshot.hdf5
Output:
  filepath: ./
  basename: test.hdf5
```

Multi-scale production:
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
  cdim: 200
Tree:
  max_leaf_count: 150
Input:
  filepath: /data/snap_0000.hdf5
  placeholder: "0000"
Output:
  filepath: /output/
  basename: gridded_0000.hdf5
  write_masses: 1
```

## See Also

- **[Gridding](gridding.md)** - Detailed grid type descriptions
- **[Runtime Arguments](runtime-arguments.md)** - Command line options
- **[Quickstart](quickstart.md)** - Get started quickly
