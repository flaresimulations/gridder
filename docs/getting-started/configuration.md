# Configuration

## Environment Variables

### OpenMP Thread Control

```bash
export OMP_NUM_THREADS=8  # Number of OpenMP threads
```

### MPI Configuration

```bash
export OMP_NUM_THREADS=4
mpirun -n 2 ./build_mpi/parent_gridder params.yml 1
# Total cores = 2 ranks × 4 threads = 8 cores
```

## Recommended Configurations

### Single-Node Workstation (16 cores)

```bash
export OMP_NUM_THREADS=16
./build/parent_gridder params.yml 1
```

### HPC Cluster Node (48 cores)

```bash
# Option 1: Pure OpenMP
export OMP_NUM_THREADS=48
./build/parent_gridder params.yml 1

# Option 2: Hybrid MPI+OpenMP (if grid is very large)
export OMP_NUM_THREADS=12
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1
```
