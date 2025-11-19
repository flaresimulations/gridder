# Quickstart

## Minimal Example

Create `params.yml`:

```yaml
Kernels:
  nkernels: 2
  kernel_radius_1: 0.5  # Mpc/h
  kernel_radius_2: 1.0  # Mpc/h

Grid:
  type: uniform
  cdim: 50  # 50^3 = 125,000 grid points

Tree:
  max_leaf_count: 200

Input:
  filepath: /path/to/snapshot_0042.hdf5

Output:
  filepath: ./output/
  basename: gridded_snapshot.hdf5
  write_masses: 0
```

Run:

```bash
# Single-node
./build/parent_gridder params.yml 8

# Multi-node (MPI)
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```

Output structure:
```
/Grids/Kernel_0/GridPointOverDensities
/Grids/Kernel_1/GridPointOverDensities
/Grids/GridPointPositions
```

## Next Steps

- [Installation](installation.md) - Build instructions
- [Parameters](parameters.md) - Full parameter reference
- [Gridding](gridding.md) - Grid types
- [MPI](mpi.md) - Multi-node execution
