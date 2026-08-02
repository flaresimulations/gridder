# Performance and Memory Model

This page describes the current serial/OpenMP implementation. Timings are
machine- and filesystem-dependent; use the phase timers printed by the gridder
when comparing runs.

## Particle Storage

Particle properties are owned once by `Simulation` in field-separated arrays:

- positions are a flat, particle-major `x, y, z` array;
- masses are stored in a separate array; and
- cells store 64-bit indices into those arrays rather than particle objects.

Masses are retained per particle. The gridder does not assume that all
particles have the same mass.

During octree construction, a parent counts the particles assigned to each
child before allocating the child index arrays. This gives the children exact
capacities. Once a cell has split, its particle-index array is released;
particle indices are therefore retained only by leaves. Internal nodes keep
aggregate particle counts and masses so that a fully enclosed node can still
be accepted without visiting its leaves. The eight siblings created by a split
share one contiguous allocation to reduce allocator traffic and improve tree
traversal locality.

The configured `Tree/max_leaf_count` trades tree depth and construction cost
against the number of particles inspected in a leaf. Its best value depends on
the particle distribution and kernel radii.

## Kernel Traversal

All configured radii are evaluated in one octree traversal per grid point.
Radii are sorted internally for the search, but results remain associated with
their original configuration and output order. A particle distance is computed
once. Contributions to a contiguous range of sorted radii are recorded as two
difference updates and finalized with one prefix pass, rather than updating
every containing radius individually. Exact periodic point-to-cell AABB bounds
reject cells outside a kernel and accept fully enclosed cells, which contribute
their aggregate count and exact aggregate mass.

OpenMP distributes useful top-level cells between threads. Each thread writes
only the accumulators belonging to grid points owned by its top-level cell;
neighboring particle trees are read-only. Recursive work inside one top-level
tree is serial.

## Parallel and Serial Phases

The main OpenMP-parallel phases are:

- top-level cell and neighbor construction;
- full-read particle-index filling and cell-mass accumulation;
- particle-to-cell validation and in-place compaction;
- useful-cell marking and non-useful-cell cleanup;
- top-level octree construction; and
- fused kernel traversal.

Grid-point generation and useful-cell marking currently run on one thread.
Sparse HDF5 transfers remain serial, while their selected cell ranges are
planned in file-offset order. Grid-point assignment uses per-thread counts and
disjoint insertion ranges instead of a global critical section. Serial-output
buffers are assembled with OpenMP before one serial write per dataset.

## HDF5 I/O

Both normal and MPI builds use the serial HDF5 API. OpenMP threads do not issue
concurrent calls through one `HDF5Helper`. This avoids relying on a thread-safe
HDF5 build and prevents unverified contention on a shared filesystem.

When most cells are useful, the full-read path performs two large blocking
reads in sequence: `PartType1/Masses`, followed by
`PartType1/Coordinates`. The resulting vectors are moved into `Simulation`, so
there is no additional full-array copy. OpenMP cannot accelerate the time spent
inside these blocking reads. Potential I/O changes must be benchmarked on the
target filesystem; concurrent reads are not automatically faster and require
an HDF5 build that supports the intended access pattern.

For spatially restricted runs, the sparse path sorts selected cells by file
offset, merges nearby file ranges, and reads only useful particle chunks.
`Input/part_gap_fill_fraction` controls the trade-off between extra particles
read and the number of HDF5 operations. This path is serial and incurs two
reads per merged chunk, one for each particle property dataset. Both datasets
remain open across the chunk loop to avoid repeated metadata operations.

Serial output keeps HDF5 calls on one thread, but assembles cell-grouped arrays
in parallel and writes each complete dataset once. For `K` kernels, a release
build makes `3 + K * (2 + M)` dataset writes, where `M` is one when mass output
is enabled and zero otherwise. Grid-point starts, grid-point counts, and kernel
particle counts use signed 64-bit output datasets. Output logs report buffer
assembly and actual dataset-write time separately.

MPI ranks write separate serial HDF5 files. Rank 0 subsequently combines these
files as described in the [MPI guide](mpi.md); the code does not use parallel
HDF5 collective I/O.

## Reference Profile

One 5.832-billion-particle FLAMINGO run on the optimized indexed-tree branch
produced the following profile:

| Phase | Time (s) | Fraction of total |
|---|---:|---:|
| HDF5 particle read | 659.024 | 74.8% |
| Fused kernel traversal | 113.255 | 12.9% |
| Octree splitting | 30.756 | 3.5% |
| Particle indexing | 25.610 | 2.9% |
| Serial output | 25.392 | 2.9% |
| Validation and movement | 25.093 | 2.8% |
| Other work | 1.582 | 0.2% |
| **Total** | **880.712** | **100%** |

The input comprised 173.81 GiB and achieved about 270 MiB/s in that run.
Observed read times on the same problem have varied from roughly 521 to 659
seconds, so total runtime alone is not a reliable measure of a compute change.
Compare individual phases and repeat I/O-bound measurements.

After excluding input I/O, kernel traversal accounts for approximately half of
the remaining runtime. This gives the following practical priority order:

1. improve or hide input throughput only after profiling the target HDF5 build
   and filesystem;
2. measure the fused-search range-accumulation and exact AABB pruning changes,
   especially when many radii are configured;
3. improve octree allocation locality; and
4. parallelize remaining serial preparation only when profiles justify it.

Any optimization must preserve periodic wrapping, variable particle masses,
inclusive kernel boundaries, configured radius/output order, and correction of
particles whose coordinates disagree with their input cell metadata.

## Profiling Guidance

- Use a Release build; Debug enables expensive correctness checks.
- Keep the thread count, NUMA placement, input file, and filesystem cache state
  consistent between comparisons.
- Compare the gridder's named phase timers rather than only wall-clock time.
- Measure HDF5 changes on the production filesystem and repeat runs because
  cache and shared-filesystem load can dominate results.
- Start with representative kernel counts and radii: larger radii visit more
  tree nodes, while more radii increase accumulator work inside the fused
  traversal.
