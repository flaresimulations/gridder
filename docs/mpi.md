# MPI Parallelization

Hybrid MPI + OpenMP for multi-node execution.

**When to use:**
- Simulations > 10 GB
- Grid points > 10M
- More than 16 cores needed

## Build

```bash
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi
```

## Usage

```bash
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1 42
# Total: 4 ranks × 2 threads = 8 cores
```

SLURM example:
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun ./build_mpi/parent_gridder params.yml 1 ${SNAPSHOT}
```

## Workflow

Domain decomposition using space-filling curve assigns cells to ranks for load balancing.

**Execution:**
1. Rank 0 reads metadata, broadcasts to all
2. Each rank loads particles for its cells
3. Ghost cell exchange near boundaries
4. Build octree per rank
5. Compute overdensities
6. Write per-rank HDF5 files
7. Rank 0 creates virtual file combining all ranks

**Ghost cells:** Ranks exchange particles within `max_kernel_radius` of boundaries for accurate kernel calculations.

## Output

Per-rank files: `gridded_data_rank{0,1,2,...}.hdf5`
Virtual file: `gridded_data.hdf5` (combines all ranks, no duplication)

Read as single file:
```python
with h5py.File('gridded_data.hdf5', 'r') as f:
    data = f['Grids/Kernel_0/GridPointOverDensities'][:]
```

## Performance

**Rank selection:**
- Minimum: ~100k grid points/rank
- Optimal: 500k - 5M grid points/rank
- Maximum: ~128-256 ranks (communication overhead)

Quick formula: `N_ranks ≈ GridPoints / 500k` or `SimulationGB / 5`

**Hybrid config:**
- Balanced: 4-8 OpenMP threads per rank (usually optimal)
- Test different configurations for your system

**Memory per rank:** `Simulation/N_ranks + overhead`
- Overhead: ~1.5× particle data (octree, grid, ghosts)
- Example: 40 GB sim ÷ 8 ranks = ~7 GB/rank total

## Troubleshooting

??? failure "MPI_Init failed"

    **Error:**
    ```
    ORTE_ERROR_LOG: Not found
    ```

    **Cause:** MPI not properly initialized.

    **Solution:**

    1. Ensure MPI module is loaded:
       ```bash
       module load openmpi  # Or your MPI flavor
       ```

    2. Use `mpirun` or `srun`, not direct execution:
       ```bash
       # Wrong:
       ./build_mpi/parent_gridder params.yml 8

       # Right:
       mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
       ```

??? failure "Ranks hanging at MPI_Barrier"

    **Symptom:** Program freezes, no output.

    **Cause:** Deadlock from mismatched MPI calls or errors on some ranks.

    **Debug:**

    ```bash
    # Run with high verbosity
    export OMP_NUM_THREADS=1
    mpirun -n 2 ./parent_gridder params.yml 1 0 2  # verbosity=2
    ```

    **Common causes:**

    - One rank encounters error, others wait forever
    - File I/O fails on some nodes
    - Different parameter files on different ranks

    **Solution:** Ensure all ranks can access all files.

??? failure "Output files not created"

    **Symptom:** Some rank files missing.

    **Cause:** Rank failed but didn't propagate error.

    **Debug:**

    ```bash
    # Check stderr from all ranks
    mpirun -n 4 ./parent_gridder params.yml 1 2>&1 | tee output.log
    grep -i error output.log
    ```

??? failure "Poor scaling / No speedup"

    **Symptom:** 4 ranks barely faster than 1 rank.

    **Causes:**

    1. **Too few grid points per rank:**
       ```yaml
       # Bad: 10k points ÷ 16 ranks = 625 points/rank
       Grid:
         cdim: 22  # Only 10,648 points total
       ```

       **Fix:** Use fewer ranks or more grid points.

    2. **Large kernels causing ghost overhead:**

       Check ghost cell fraction in output:
       ```
       [INFO] Ghost cells: 45% of total
       ```

       If >30%, consider:
       - Smaller kernels
       - Fewer ranks
       - Larger cells (higher `max_leaf_count`)

    3. **I/O bottleneck:**

       All ranks reading same file simultaneously. Use:
       - Lustre striping on HPC
       - Parallel filesystem (not NFS)

??? question "How many ranks should I use?"

    **Quick formula:**

    ```
    N_ranks ≈ GridPoints / 500,000
    N_ranks ≈ SimulationGB / 5
    ```

    Use the smaller of the two.

    **Example:**

    - 2M grid points, 20 GB simulation
    - By points: 2M / 500k = 4 ranks
    - By memory: 20 / 5 = 4 ranks
    - **Use:** 4 ranks

## Advanced MPI

### Custom MPI Launcher

Some systems use different launchers:

```bash
# OpenMPI
mpirun -n 4 ./parent_gridder params.yml 1

# Intel MPI
mpiexec -n 4 ./parent_gridder params.yml 1

# SLURM
srun -n 4 ./parent_gridder params.yml 1

# PBS/Torque
mpirun -np 4 ./parent_gridder params.yml 1
```

### Process Binding

**Bind ranks to cores:**

```bash
# OpenMPI
mpirun -n 4 --bind-to core ./parent_gridder params.yml 1

# With rank-to-core mapping
mpirun -n 4 --map-by core --bind-to core ./parent_gridder params.yml 1
```

**Benefits:**

- Prevents thread migration (better cache performance)
- Avoids hyperthreading (use physical cores)

### Network Optimization

**InfiniBand:**

```bash
# Force use of InfiniBand
export OMPI_MCA_btl=self,openib
mpirun -n 16 ./parent_gridder params.yml 1
```

**Ethernet:**

```bash
# Use TCP for communication
export OMPI_MCA_btl=self,tcp
mpirun -n 8 ./parent_gridder params.yml 1
```

### Debugging MPI Issues

**Verbose MPI output:**

```bash
# OpenMPI debug info
export OMPI_MCA_btl_base_verbose=30
mpirun -n 2 ./parent_gridder params.yml 1
```

**Attach debugger:**

```bash
# GDB on rank 0
mpirun -n 4 xterm -e gdb ./parent_gridder
```

**Valgrind with MPI:**

```bash
mpirun -n 2 valgrind --leak-check=full ./parent_gridder params.yml 1
```

## Performance Benchmarks

### Example: 50 Mpc/h Box, 100M Particles

| Configuration | Grid Points | Time | Memory/Rank |
|---------------|-------------|------|-------------|
| 1 rank × 16 threads | 1M | 180s | 8 GB |
| 4 ranks × 4 threads | 1M | 52s | 2.5 GB |
| 8 ranks × 2 threads | 1M | 30s | 1.5 GB |
| 16 ranks × 1 thread | 1M | 22s | 1 GB |

**Speedup:** ~8× from 1 to 16 ranks (good strong scaling)

### Example: Weak Scaling Test

| Ranks | Grid Points | Particles | Time | Efficiency |
|-------|-------------|-----------|------|------------|
| 1 | 1M | 100M | 180s | 100% |
| 2 | 2M | 200M | 190s | 95% |
| 4 | 4M | 400M | 200s | 90% |
| 8 | 8M | 800M | 220s | 82% |

**Result:** Good weak scaling up to 8 ranks, then communication overhead grows.

## See Also

- **[Installation](installation.md#multi-node-build-mpi--openmp-hybrid)** - Building with MPI
- **[Runtime Arguments](runtime-arguments.md)** - Command line options
- **[Parameters](parameters.md)** - Configuration reference
- **[Quickstart](quickstart.md)** - Basic examples
