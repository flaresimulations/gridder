# Runtime Arguments

```text
parent_gridder <parameter_file> <nthreads> [snapshot_number] [verbosity]
```

## Arguments

### `parameter_file`

Required readable `.yml` or `.yaml` parameter file. Other extensions produce a
warning.

### `nthreads`

Required positive OpenMP thread count. This argument controls the thread count
in both serial and MPI builds.

### `snapshot_number`

Optional non-negative integer used for placeholder replacement. Default: `0`.
It is zero-padded to the length of `Input/placeholder` and applied to
`Input/filepath`, `Output/basename`, and `Grid/grid_file`.

### `verbosity`

Optional logging level. Default: `1`.

| Value | Output |
|-------|--------|
| `0` | Errors only |
| `1` | Normal messages from rank 0 |
| `2` | Messages from every rank |

## Examples

```bash
# Eight threads, snapshot number defaults to zero
./build/parent_gridder params.yml 8

# Snapshot 42 with normal logging
./build/parent_gridder params.yml 8 42 1

# Four MPI ranks, two threads per rank
mpirun -n 4 ./build_mpi/parent_gridder params.yml 2 42
```

`OMP_PROC_BIND` and `OMP_PLACES` may be used for thread affinity, but the
positional `nthreads` argument sets the team size.

## Help

```bash
./build/parent_gridder --help
```

The supported help options are `-h` and `--help`. There is no `--version`
option; build metadata is printed during a normal run.
