# MPI Parallelization

## Overview

The gridder supports MPI for distributed-memory parallelization across multiple nodes. This is combined with OpenMP for a hybrid MPI+OpenMP approach.

## Build Configuration

### Enabling MPI

```bash
cmake -B build_mpi -DENABLE_MPI=ON
cmake --build build_mpi
```

### Verify MPI Support

```bash
# Check for MPI linkage
ldd build_mpi/parent_gridder | grep mpi  # Linux
otool -L build_mpi/parent_gridder | grep mpi  # macOS
```

## Domain Decomposition

The gridder uses **spatial domain decomposition** to distribute cells across MPI ranks:

1. Top-level cells are created and distributed across ranks
2. Each rank owns a subset of cells based on spatial partitioning
3. Ghost cells ("proxy cells") are exchanged at partition boundaries
4. Each rank processes its domain independently
5. Output is written to separate HDF5 files per rank

### Partition Strategy

```cpp
// Cells are assigned to ranks based on spatial location
rank = (cell_x * n_ranks_y * n_ranks_z +
        cell_y * n_ranks_z +
        cell_z) % n_ranks
```

## Ghost Cell Exchange

Cells at partition boundaries require particle data from neighboring ranks:

```cpp
// Proxy cells: cells owned by other ranks but needed for kernel overlap
if (distance_to_boundary < max_kernel_radius) {
  mark_as_proxy_cell();
  exchange_particles_with_neighbor_rank();
}
```

### Exchange Protocol

1. **Flag phase**: Each rank identifies proxy cells needed from neighbors
2. **Request phase**: Send cell IDs to neighbor ranks
3. **Response phase**: Neighbor ranks send particle data
4. **Integration phase**: Received particles used for gridding

## Running with MPI

### Basic Usage

```bash
# 4 MPI ranks, 8 OpenMP threads per rank = 32 total cores
export OMP_NUM_THREADS=8
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```

### Recommended Configurations

#### Small Cluster (16 cores per node, 4 nodes = 64 cores)

```bash
# Option 1: 4 ranks × 16 threads
export OMP_NUM_THREADS=16
mpirun -n 4 --map-by node ./build_mpi/parent_gridder params.yml 1

# Option 2: 8 ranks × 8 threads (better load balancing)
export OMP_NUM_THREADS=8
mpirun -n 8 --map-by node:PE=8 ./build_mpi/parent_gridder params.yml 1
```

#### Large HPC System (48 cores per node, 16 nodes = 768 cores)

```bash
# 16 ranks × 48 threads (one rank per node)
export OMP_NUM_THREADS=48
mpirun -n 16 --map-by node ./build_mpi/parent_gridder params.yml 1
```

### Process Binding

For NUMA systems, ensure proper binding:

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=24
mpirun -n 4 --bind-to socket --map-by socket:PE=24 \
  ./build_mpi/parent_gridder params.yml 1
```

## Performance Characteristics

### Scaling Behavior

| Ranks | Threads/Rank | Total Cores | Speedup | Efficiency |
|-------|--------------|-------------|---------|------------|
| 1 | 8 | 8 | 1.0× | 100% |
| 2 | 8 | 16 | 1.8× | 90% |
| 4 | 8 | 32 | 3.4× | 85% |
| 8 | 8 | 64 | 6.2× | 78% |

*Typical results for 100³ grid with 10M particles*

### Bottlenecks

1. **Ghost cell exchange**: Communication overhead at partition boundaries
2. **Load imbalance**: Non-uniform particle distribution
3. **I/O serialization**: Each rank reads from shared HDF5 file sequentially

## Output Files

Each MPI rank writes a separate output file:

```
output/gridded_data_rank_0.hdf5
output/gridded_data_rank_1.hdf5
output/gridded_data_rank_2.hdf5
output/gridded_data_rank_3.hdf5
```

### Merging Output (Optional)

While not required, you can merge rank files for convenience:

```python
import h5py
import numpy as np

# Example: Concatenate grid points from all ranks
all_coords = []
all_overdensities = {}

for rank in range(n_ranks):
    with h5py.File(f'gridded_data_rank_{rank}.hdf5', 'r') as f:
        all_coords.append(f['Grids/GridPointCoordinates'][:])

        for kernel in f['Grids'].keys():
            if kernel.startswith('Kernel_'):
                if kernel not in all_overdensities:
                    all_overdensities[kernel] = []
                all_overdensities[kernel].append(
                    f[f'Grids/{kernel}/GridPointOverDensities'][:]
                )

coords = np.concatenate(all_coords)
for kernel, data in all_overdensities.items():
    all_overdensities[kernel] = np.concatenate(data)
```

## Troubleshooting

### MPI Hangs at Startup

Check that all ranks can access the HDF5 file:

```bash
# Test file access from all nodes
mpirun -n 4 ls -l /path/to/snapshot.hdf5
```

### Memory Exhaustion

Reduce particles per rank:
- Increase number of ranks
- Reduce `max_leaf_count` to limit tree depth
- Process fewer kernel radii per run

### Load Imbalance

Monitor rank workload:

```bash
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1 2>&1 | grep "particles processed"
```

If ranks show significantly different particle counts, the domain decomposition may be unbalanced due to clustered particle distribution.

## See Also

- [OpenMP Threading](openmp.md) - Thread-level parallelization
- [Configuration Guide](../getting-started/configuration.md) - Environment setup
- [Parameter Reference](../getting-started/parameters.md) - YAML configuration options
