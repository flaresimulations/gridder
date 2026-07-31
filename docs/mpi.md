# MPI Execution

The MPI build distributes top-level cells across ranks and uses OpenMP within
each rank.

## Build

```bash
cmake -B build_mpi -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi
```

## Run

```bash
# Four ranks, two OpenMP threads per rank
mpirun -n 4 ./build_mpi/parent_gridder params.yml 2
```

The second positional argument sets threads per rank. Choose rank and thread
counts that fit the allocated hardware. Affinity can be controlled separately:

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
mpirun -n 4 ./build_mpi/parent_gridder params.yml 2
```

## Decomposition

Top-level cells are visited in cell-ID order and assigned in contiguous runs,
targeting approximately equal particle counts per rank. Boundary cells needed
by another rank are exchanged as proxy cells before kernel calculations.

This is a particle-balanced contiguous decomposition. It is not a Hilbert or
other space-filling-curve partitioner.

## Particle Input

Each rank reads particle ranges for useful local cells. Nearby ranges may be
merged to reduce HDF5 calls. Configure this trade-off with
`Input/part_gap_fill_fraction`; see [Parameters](parameters.md).

## Output

For an output basename `grid.hdf5`, MPI writes files such as:

```text
grid_rank0.hdf5
grid_rank1.hdf5
grid_rank2.hdf5
grid_rank3.hdf5
grid.hdf5
```

The final `grid.hdf5` is an ordinary combined HDF5 file. Rank 0 reads the rank
files and copies their datasets into it. It is not an HDF5 virtual dataset and
therefore duplicates the combined data.

Keep the rank files until combination has completed successfully. The combined
file can then be read like serial output.

## Batch Example

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4

export OMP_PROC_BIND=close
export OMP_PLACES=cores
srun ./build_mpi/parent_gridder params.yml 4 42
```

## Troubleshooting

- Confirm the executable was configured with `-DENABLE_MPI=ON`.
- Ensure ranks multiplied by threads per rank does not exceed the allocation.
- Use verbosity `2` with a small rank count to inspect rank-specific progress.
- If memory is exhausted, use more ranks or fewer grid points per run.
