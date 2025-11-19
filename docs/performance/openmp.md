# OpenMP Threading

## Overview

Both build configurations use OpenMP for intra-node parallelization:

- **Single-node build**: OpenMP only (multi-threaded)
- **Multi-node build**: MPI + OpenMP hybrid

## Current Parallelization

### Position Unpacking (src/cell.cpp:593)

Parallel unpacking of 3D position arrays from flat HDF5 data:

```cpp
#pragma omp parallel for schedule(static)
for (size_t p = 0; p < chunk.particle_count; p++) {
  chunk.positions[p] = {pos_flat[p * 3], pos_flat[p * 3 + 1],
                        pos_flat[p * 3 + 2]};
}
```

**Speedup**: ~T× where T = number of threads  
**Applies to**: Both serial and MPI builds

## Performance Tuning

### Thread Count Selection

```bash
# Auto-detect (recommended for single-node)
./build/parent_gridder params.yml 1

# Manual control
export OMP_NUM_THREADS=8
./build/parent_gridder params.yml 1
```

### Binding Strategy

For NUMA systems:

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8
./build/parent_gridder params.yml 1
```

## Expected Performance

| Component | Parallelization | Speedup (8 threads) |
|-----------|----------------|---------------------|
| HDF5 I/O | Sequential | 1× (bottleneck) |
| Position unpacking | OpenMP | ~8× |
| Cell processing | Sequential | 1× |
| **Overall** | **Hybrid** | **~1.1-1.2×** |

The modest overall speedup is due to Amdahl's Law: HDF5 I/O dominates (~60-80% of time) and cannot be parallelized with standard HDF5.
