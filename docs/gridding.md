# Grid Types

Grid points are the locations where enclosed mass and overdensity are
calculated for every configured kernel radius.

## Uniform Grid

```yaml
Grid:
  type: uniform
  cdim: 100
```

This creates `cdim^3` regularly spaced points across the simulation box.
`Grid/type` defaults to `uniform`, but `Grid/cdim` remains required.

Uniform grids are convenient for maps and FFT-based downstream analysis. Their
memory cost grows cubically with `cdim`.

## Random Grid

```yaml
Grid:
  type: random
  n_grid_points: 1000000
  random_seed: 42
```

`Grid/n_grid_points` is required. `Grid/random_seed` defaults to `42`, allowing
runs to be reproduced. MPI code paths that generate points per rank add the
rank to the configured seed.

Random grids provide a requested sample count without constraining it to a
cube, but they do not form a regular image grid.

## File-Based Grid

```yaml
Grid:
  type: file
  grid_file: /data/grid_points_0000.txt
```

The file contains three whitespace-separated coordinates per line:

```text
# x y z
5.0 5.0 5.0
10.2 15.3 20.1
25.0 30.0 35.0
```

Blank lines and lines beginning with `#` are ignored. Coordinates use the same
comoving units as the snapshot and should lie inside its box. The snapshot
placeholder is replaced in `Grid/grid_file` in the same way as input and output
paths.

File grids are useful for halo centers, selected structures, and reusing an
identical sample across snapshots.

## Kernel Configuration

Every point is evaluated for each configured radius:

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0
```

Radii use the same comoving length units as coordinates. Larger radii generally
visit more particles and therefore require more work. The gridder evaluates all
configured radii in one fused octree traversal and preserves their configured
order in the output. See [Performance and Memory Model](performance.md) for the
search and storage design.

## Choosing A Grid

| Goal | Grid type |
|------|-----------|
| Regular map or volume | `uniform` |
| Fixed-size statistical sample | `random` |
| Selected external positions | `file` |

Start with a small point count and representative kernel radii when validating
a new snapshot. Increase the resolution only after checking output and memory
use.
