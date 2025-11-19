# Gridder Tools

Utility scripts for working with the parent_gridder.

## convert_to_gridder_format.py

Convert HDF5 simulation snapshots with arbitrary key names to the format expected by parent_gridder.

### Requirements

```bash
pip install h5py numpy
pip install mpi4py  # Optional, for MPI support
```

### Usage

#### Serial Mode

Convert a single file:

```bash
python tools/convert_to_gridder_format.py input.hdf5 output.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

#### MPI Mode

For large files, use MPI to process in parallel. Each rank writes a separate file, and a virtual HDF5 file is created to provide a unified view:

```bash
mpirun -np 4 python tools/convert_to_gridder_format.py input.hdf5 output.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

This creates:
- `output_rank_0.hdf5`, `output_rank_1.hdf5`, etc. (per-rank files)
- `output.hdf5` (virtual file that combines all ranks)

The gridder can read either the virtual file or individual rank files.

### Options

- `--coordinates-key KEY`: HDF5 path to particle coordinates (required)
- `--masses-key KEY`: HDF5 path to particle masses (required)
- `--particle-type TYPE`: Output particle type group name (default: `PartType1`)
- `--copy-header`: Copy Header group from input to output
- `--header-key KEY`: Input header group path (default: `Header`)

### Examples

#### Convert SWIFT snapshot

```bash
python tools/convert_to_gridder_format.py \
    snapshot_0100.hdf5 converted_0100.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

#### Convert with custom keys

```bash
python tools/convert_to_gridder_format.py \
    my_simulation.hdf5 gridder_input.hdf5 \
    --coordinates-key DarkMatter/Positions \
    --masses-key DarkMatter/ParticleMasses \
    --particle-type PartType1
```

#### Large file with MPI

```bash
mpirun -np 8 python tools/convert_to_gridder_format.py \
    large_snapshot.hdf5 converted.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

### Output Format

The script creates HDF5 files with the following structure:

```
output.hdf5
├── Header/           (if --copy-header specified)
└── PartType1/        (or custom name from --particle-type)
    ├── Coordinates   (shape: [N, 3], dtype: float64)
    └── Masses        (shape: [N], dtype: float64)
```

In MPI mode, the virtual file provides a transparent view of the combined data from all rank files.

### Notes

- **Compression**: Output files use gzip compression (level 4) to reduce size
- **Virtual files**: Require HDF5 1.10+ and `libver='latest'`
- **Memory**: Each rank processes approximately `N/nranks` particles in MPI mode
- **Particle ordering**: Preserved from input file within each rank
