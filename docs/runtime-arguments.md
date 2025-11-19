# Runtime Arguments

```bash
parent_gridder <parameter_file> <nthreads> [snapshot_number] [verbosity]
```

## Arguments

### `parameter_file`

**Type:** String (file path)
**Required • Position 1**

Path to YAML parameter file. See [Parameters](parameters.md).

### `nthreads`

**Type:** Integer
**Required • Position 2**

Number of OpenMP threads. Recommend physical core count (not hyperthreads).

In MPI mode, use `OMP_NUM_THREADS` environment variable instead:
```bash
export OMP_NUM_THREADS=2
mpirun -n 4 ./parent_gridder params.yml 1
```

### `snapshot_number`

**Type:** Integer (≥ 0)
**Optional • Position 3**

Snapshot number for placeholder replacement in file paths (`Input/filepath`, `Output/basename`, `Grid/grid_file`).

Example:
```bash
./parent_gridder params.yml 8 42  # Replaces "0000" with "0042"
```

Zero-padded to match placeholder length.

### `verbosity`

**Type:** Integer (0, 1, or 2)
**Optional • Position 4 (default: 1)**

Output verbosity:
- **0**: Errors only
- **1**: Rank 0 only (default)
- **2**: All ranks (debug mode, use with <10 ranks)

## Examples

Basic:
```bash
./parent_gridder params.yml 8
./parent_gridder params.yml 8 42
./parent_gridder params.yml 8 42 0
```

MPI:
```bash
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1 42
```

Batch script:
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2

export OMP_NUM_THREADS=2
mpirun ./parent_gridder params.yml 1 ${SNAP_NUM}
```

## Help

```bash
./parent_gridder --help
./parent_gridder --version
```

## Environment Variables

- `OMP_NUM_THREADS`: OpenMP thread count (use instead of nthreads arg in MPI mode)
- `OMP_PROC_BIND=close`: Thread affinity (recommended)
- `OMP_PLACES=cores`: Bind to physical cores (recommended)

## See Also

- **[Parameters](parameters.md)** - Parameter file configuration
- **[MPI](mpi.md)** - Multi-node execution details
- **[Quickstart](quickstart.md)** - Getting started guide
