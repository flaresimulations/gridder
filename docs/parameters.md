# Parameter Reference

Parameter files use simple indentation-based YAML-style mappings. The parser
supports the scalar mappings shown below; it is not a complete YAML parser.

## Complete Example

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0

Grid:
  type: random
  n_grid_points: 1000000
  random_seed: 42

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: /data/snapshot_0000.hdf5
  placeholder: "0000"
  part_gap_fill_fraction: 0.01

Output:
  filepath: /data/grids/
  basename: grid_0000.hdf5
  write_masses: 1
```

## Kernels

### `Kernels/nkernels`

Required integer giving the number of spherical top-hat kernels.

### `Kernels/kernel_radius_N`

Required floating-point radius for each kernel numbered from `1` through
`nkernels`. Radii use the same comoving length units as particle coordinates.

## Grid

### `Grid/type`

One of `uniform`, `random`, or `file`. Default: `uniform`.

### Uniform Grid

`Grid/cdim` is required and creates `cdim^3` regularly spaced points.

### Random Grid

`Grid/n_grid_points` is required. `Grid/random_seed` controls reproducibility
and defaults to `42`. In MPI paths that generate points per rank, the rank is
added to the seed.

### File Grid

`Grid/grid_file` gives a text file containing three whitespace-separated
coordinates per line. Blank lines and lines beginning with `#` are ignored.
Snapshot placeholder replacement is applied to this path.

## Cosmology

The following values are required:

- `Cosmology/h`: reduced Hubble parameter
- `Cosmology/Omega_cdm`: cold dark matter density parameter
- `Cosmology/Omega_b`: baryon density parameter

The gridder uses `Omega_cdm + Omega_b` to calculate the mean comoving matter
density. These values come from the parameter file, not an HDF5 cosmology
group.

## Tree

### `Tree/max_leaf_count`

Maximum particles in an octree leaf before subdivision. Default: `200`.

## Input

### `Input/filepath`

Required path to the cell-indexed HDF5 snapshot.

### `Input/placeholder`

String replaced by the zero-padded command-line snapshot number in input,
output, and grid-file paths. Default: `"0000"`. The command-line snapshot
number defaults to `0`, so replacement still occurs when argument 3 is omitted.

### `Input/part_gap_fill_fraction`

Controls merging of nearby particle read ranges. A gap containing less than
this fraction of the total particle count is included in a neighboring read to
reduce HDF5 operations. Default: `0.01`. Larger values can read more unneeded
particles while issuing fewer reads.

## Output

### `Output/filepath`

Required output directory. Missing directories are created recursively.

### `Output/basename`

Required output filename. Snapshot placeholder replacement is supported.

### `Output/write_masses`

Set to `1` to write enclosed masses as well as overdensities. Default: `0`.

## Input HDF5 Requirements

The gridder reads:

```text
Header attributes: Redshift, NumPart_Total, BoxSize
Cells/Meta-data attributes: dimension, size
Cells/Counts/PartType1
Cells/OffsetsInFile/PartType1
PartType1/Coordinates
PartType1/Masses
```

See [Snapshot conversion](conversion.md) for constructing the cell index.
