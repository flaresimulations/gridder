# Runtime Arguments

Command line arguments for controlling gridder execution.

## Synopsis

```bash
parent_gridder <parameter_file> <nthreads> [snapshot_number] [verbosity]
```

## Arguments

### `parameter_file`

!!! info "Required • Position 1"

**Type:** String (file path)
**Description:** Path to YAML parameter configuration file

**Example:**
```bash
./parent_gridder params.yml 8
./parent_gridder /path/to/config/my_params.yml 4
```

**Notes:**

- Must be a valid YAML file
- See [Parameters](parameters.md) for file format
- Relative or absolute paths supported

### `nthreads`

!!! info "Required • Position 2"

**Type:** Integer
**Range:** 1 to hardware limit
**Description:** Number of OpenMP threads to use

**Example:**
```bash
./parent_gridder params.yml 8   # Use 8 OpenMP threads
./parent_gridder params.yml 16  # Use 16 OpenMP threads
```

**Recommendations:**

=== "Single-Node Build"

    ```bash
    # Use number of physical cores (not hyperthreads)
    # Check with: lscpu | grep "^CPU(s):"
    ./parent_gridder params.yml 8
    ```

=== "MPI Build"

    ```bash
    # Threads per rank × ranks = total cores
    export OMP_NUM_THREADS=2
    mpirun -n 4 ./parent_gridder params.yml 1
    # Note: nthreads argument ignored in MPI mode
    # Use OMP_NUM_THREADS environment variable instead
    ```

**Performance Notes:**

- More threads ≠ always faster
- Sweet spot: Number of physical cores
- Hyperthreading (2× cores) usually gives <20% speedup
- Measure performance for your specific case

**Auto-Detection:**

The gridder will auto-detect available cores, but explicit specification is recommended for reproducibility:

```bash
# Let OpenMP choose (may use all cores including hyperthreads)
./parent_gridder params.yml 1

# Better: Explicitly set to physical core count
./parent_gridder params.yml $(lscpu | grep "^Core(s)" | awk '{print $4}')
```

### `snapshot_number`

!!! info "Optional • Position 3"

**Type:** Integer
**Range:** ≥ 0
**Default:** None (no replacement)
**Description:** Snapshot number for placeholder replacement in file paths

**Example:**
```bash
# Process snapshot 42
./parent_gridder params.yml 8 42
```

**Behavior:**

When provided, replaces all occurrences of the placeholder string in:

- `Input/filepath`
- `Output/basename`
- `Grid/grid_file` (if `type: file`)

**Parameter File:**
```yaml
Input:
  filepath: /data/snapshot_0000.hdf5
  placeholder: "0000"

Output:
  basename: grid_0000.hdf5
```

**Command:**
```bash
./parent_gridder params.yml 8 42
```

**Actual Paths:**
```
Input:  /data/snapshot_0042.hdf5
Output: grid_0042.hdf5
```

**Zero-Padding:**

The snapshot number is zero-padded to match placeholder length:

| Placeholder | Snapshot | Result |
|-------------|----------|--------|
| `"000"` | 5 | `005` |
| `"0000"` | 5 | `0005` |
| `"0000"` | 42 | `0042` |
| `"0000"` | 1234 | `1234` |

**No Placeholder:**

If no `Input/placeholder` is specified in parameters, this argument has no effect (paths used as-is).

### `verbosity`

!!! info "Optional • Position 4"

**Type:** Integer
**Range:** 0, 1, or 2
**Default:** 1 (rank 0 only)
**Description:** Controls output verbosity level

**Values:**

| Level | Description | Use Case |
|-------|-------------|----------|
| **0** | Errors only | Production runs, minimal logging |
| **1** | Rank 0 only (default) | Normal use, clean MPI output |
| **2** | All ranks print | Debugging MPI issues |

**Examples:**

=== "Minimal Output (0)"

    ```bash
    ./parent_gridder params.yml 8 42 0
    ```

    Output: Only errors printed
    ```
    [ERROR] Failed to open file: snapshot_0042.hdf5
    ```

=== "Default (1)"

    ```bash
    ./parent_gridder params.yml 8 42 1
    # Or omit (default):
    ./parent_gridder params.yml 8 42
    ```

    Output: Normal progress messages from rank 0 only (in MPI)
    ```
    [INFO] Reading snapshot from /data/snapshot_0042.hdf5
    [INFO] Creating 1000000 grid points
    [INFO] Building octree...
    [TIMER] Octree construction: 2.341s
    ```

=== "All Ranks (2)"

    ```bash
    mpirun -n 4 ./parent_gridder params.yml 1 42 2
    ```

    Output: All MPI ranks print (can be very verbose!)
    ```
    [RANK 0] Loading particles for local cells
    [RANK 1] Loading particles for local cells
    [RANK 2] Loading particles for local cells
    [RANK 3] Loading particles for local cells
    ...
    ```

**Serial Mode:**

In single-node builds, levels 1 and 2 are equivalent (all output printed).

**MPI Mode:**

- **Level 0:** Suppresses non-error output from all ranks
- **Level 1:** Only rank 0 prints (keeps output clean)
- **Level 2:** All ranks print (useful for debugging MPI-specific issues)

!!! warning "Verbosity 2 in MPI"
    With many ranks, verbosity=2 produces overwhelming output. Use only for debugging with small rank counts (<10).

## Full Usage Examples

### Basic Execution

```bash
# Minimal required arguments
./parent_gridder params.yml 8

# With snapshot number
./parent_gridder params.yml 8 42

# With verbosity control
./parent_gridder params.yml 8 42 0
```

### MPI Execution

```bash
# Basic MPI run (4 ranks, 2 threads each = 8 cores)
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1 42

# With custom verbosity
export OMP_NUM_THREADS=2
mpirun -n 4 ./build_mpi/parent_gridder params.yml 1 42 1

# Debugging with full output
export OMP_NUM_THREADS=1
mpirun -n 2 ./build_mpi/parent_gridder params.yml 1 42 2
```

### Production Workflow

```bash
#!/bin/bash
# Process multiple snapshots

PARAMS="production_params.yml"
NTHREADS=16

for SNAP in {0..100}; do
    echo "Processing snapshot $SNAP"
    ./parent_gridder $PARAMS $NTHREADS $SNAP 0
    if [ $? -ne 0 ]; then
        echo "ERROR: Snapshot $SNAP failed"
        exit 1
    fi
done
```

### HPC Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=gridder
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00

# Load modules
module load gcc/11.2
module load hdf5/1.12
module load openmpi/4.1

# Set OpenMP threads (matches cpus-per-task)
export OMP_NUM_THREADS=2

# 4 nodes × 16 ranks/node × 2 threads/rank = 128 cores total
mpirun ./parent_gridder params.yml 1 ${SNAP_NUM} 1
```

## Help and Version

### Display Help

```bash
./parent_gridder --help
./parent_gridder -h
```

Output:
```
FLARES-2 Parent Gridder
Version: 0.1.0
Git: abc1234 (main)
Build: Release with OpenMP + MPI

Usage: parent_gridder <parameter_file> <nthreads> [snapshot_number] [verbosity]

Arguments:
  parameter_file   Path to YAML parameter configuration file
  nthreads        Number of OpenMP threads (1-64)
  snapshot_number Optional snapshot number (≥0, replaces placeholder)
  verbosity       Optional verbosity level: 0=errors only, 1=rank 0 only (default), 2=all ranks

Examples:
  parent_gridder params.yml 8
  parent_gridder params.yml 16 42
  parent_gridder params.yml 8 0 2  # All ranks print
  mpirun -n 4 parent_gridder params.yml 8

For more information: https://github.com/flaresimulations/gridder
```

### Check Version

```bash
./parent_gridder --version
```

Output:
```
FLARES-2 Parent Gridder
Version: 0.1.0
Git commit: abc1234def567890
Branch: main
Build date: 2024-01-15
Compiler: GNU 11.2.0
MPI: Enabled (OpenMPI 4.1.2)
OpenMP: Enabled (4.5)
HDF5: 1.12.1
```

## Exit Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| **0** | Success | Normal completion |
| **1** | Error | Parameter parsing, file I/O, runtime errors |
| **2** | Invalid arguments | Wrong number of arguments, invalid values |

**Examples:**

```bash
./parent_gridder params.yml 8
echo $?  # 0 if successful, 1 if error

# Check for errors in scripts
./parent_gridder params.yml 8 42 || echo "Gridder failed!"
```

## Environment Variables

### `OMP_NUM_THREADS`

**Description:** Number of OpenMP threads (alternative to `nthreads` argument)

**Note:** In MPI builds, use this instead of the `nthreads` argument.

```bash
# Single-node: Use argument
./parent_gridder params.yml 8

# MPI: Use environment variable
export OMP_NUM_THREADS=2
mpirun -n 4 ./parent_gridder params.yml 1
```

### `OMP_PROC_BIND`

**Description:** Thread affinity policy

**Recommended:**
```bash
export OMP_PROC_BIND=close
```

Keeps threads close to parent for better cache performance.

### `OMP_PLACES`

**Description:** Where threads can run

**Recommended:**
```bash
export OMP_PLACES=cores
```

Binds threads to physical cores (not hyperthreads).

## Troubleshooting

??? failure "Invalid number of arguments"

    **Error:**
    ```
    Invalid number of arguments (1). Expected 2-4 arguments.
    ```

    **Solution:** Provide at least parameter file and nthreads:
    ```bash
    ./parent_gridder params.yml 8
    ```

??? failure "Verbosity must be 0, 1, or 2"

    **Error:**
    ```
    ERROR: Verbosity must be 0, 1, or 2 (got 3)
    ```

    **Solution:** Use valid verbosity level:
    ```bash
    ./parent_gridder params.yml 8 42 1  # Not 3
    ```

??? failure "Snapshot number is not a valid integer"

    **Error:**
    ```
    ERROR: Snapshot number is not a valid integer
    ```

    **Solution:** Ensure snapshot is a number:
    ```bash
    ./parent_gridder params.yml 8 42    # Correct
    ./parent_gridder params.yml 8 abc   # Wrong
    ```

??? question "Which nthreads should I use?"

    **Quick answer:** Number of physical cores on your system.

    **Find it:**
    ```bash
    # Linux
    lscpu | grep "^Core(s) per socket" | awk '{print $4}'

    # macOS
    sysctl -n hw.physicalcpu

    # Use it
    NCORES=$(nproc --all)
    ./parent_gridder params.yml $NCORES
    ```

## See Also

- **[Parameters](parameters.md)** - Parameter file configuration
- **[MPI](mpi.md)** - Multi-node execution details
- **[Quickstart](quickstart.md)** - Getting started guide
