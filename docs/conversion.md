# Converting Arbitrary Snapshots

The `convert_to_gridder_format.py` tool converts HDF5 simulation snapshots from arbitrary formats into the format required by the gridder. This allows you to process simulations from any code (not just SWIFT) with the FLARES-2 Gridder.

## Overview

The conversion process:

1. Reads particle coordinates and masses from arbitrary HDF5 dataset keys
2. Creates a spatial cell structure for efficient particle lookup
3. Sorts particles by cell index
4. Writes output in the standardized gridder format

The gridder requires a hierarchical cell structure for spatial indexing. The conversion script creates a regular grid of cells (default: 16×16×16 = 4,096 cells) and assigns each particle to a cell based on its position.

## Requirements

- Python 3.7+
- h5py
- numpy
- mpi4py (optional, for MPI mode)

## Basic Usage

```bash
python tools/convert_to_gridder_format.py input.hdf5 output.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

### Required Arguments

- `input_file`: Path to input HDF5 snapshot
- `output_file`: Path to output HDF5 file
- `--coordinates-key`: HDF5 dataset path for particle coordinates
- `--masses-key`: HDF5 dataset path for particle masses

### Optional Arguments

- `--copy-header`: Copy Header group from input to output (recommended)
- `--header-key`: HDF5 key for header group (default: `Header`)
- `--particle-type`: Output particle type group name (default: `PartType1`)
- `--cdim`: Number of cells per dimension for spatial indexing (default: **16**)
- `--boxsize X Y Z`: Manually specify box size (if not in Header)

## Examples

### SWIFT-like Format

If your snapshot already uses SWIFT-style keys:

```bash
python tools/convert_to_gridder_format.py \
    swift_snapshot.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header
```

### Custom Format with Non-Standard Keys

For simulations with different naming conventions:

```bash
python tools/convert_to_gridder_format.py \
    gadget_snapshot.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key DarkMatter/Positions \
    --masses-key DarkMatter/Mass \
    --copy-header \
    --cdim 32
```

### Without Header (Manual BoxSize)

If your input file doesn't have a Header group:

```bash
python tools/convert_to_gridder_format.py \
    custom_snapshot.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key Coordinates \
    --masses-key Masses \
    --boxsize 100.0 100.0 100.0 \
    --cdim 16
```

### Non-Cubic Simulation Box

For simulations with different box sizes in each dimension:

```bash
python tools/convert_to_gridder_format.py \
    noncubic_snapshot.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key Coords \
    --masses-key Mass \
    --boxsize 100.0 200.0 150.0 \
    --copy-header
```

### Fine Cell Grid for Large Simulations

For very large simulations, increase `cdim` for better spatial indexing:

```bash
python tools/convert_to_gridder_format.py \
    large_simulation.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --cdim 64 \
    --copy-header
```

## MPI Mode

For very large snapshots, use MPI to parallelize the conversion:

```bash
mpirun -np 4 python tools/convert_to_gridder_format.py \
    huge_snapshot.hdf5 \
    gridder_snapshot.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --copy-header \
    --cdim 32
```

**MPI mode creates:**
- Per-rank files: `gridder_snapshot_rank_0.hdf5`, `gridder_snapshot_rank_1.hdf5`, ...
- Virtual file: `gridder_snapshot.hdf5` (combines all ranks)

Use the virtual file (`gridder_snapshot.hdf5`) as input to the gridder.

## Output HDF5 Structure

The conversion script produces files with this structure:

```
/Header                              # Simulation metadata (if --copy-header used)
  BoxSize: [X, Y, Z]                 # Box dimensions
  NumPart_Total: [0, N, 0, 0, 0, 0] # Total particle counts
  Redshift: float                    # Redshift value

/PartType1                           # Dark matter particles
  /Coordinates                       # Particle positions (sorted by cell)
    shape: (N, 3)
    dtype: float64
  /Masses                            # Particle masses (sorted by cell)
    shape: (N,)
    dtype: float64

/Cells                               # Spatial indexing structure
  /Meta-data
    dimension: [cdim, cdim, cdim]    # Number of cells per dimension
    size: [dx, dy, dz]               # Physical size of each cell
  /Counts
    /PartType1                       # Number of particles in each cell
      shape: (cdim³,)
      dtype: int32
  /OffsetsInFile
    /PartType1                       # Starting index for particles in each cell
      shape: (cdim³,)
      dtype: int64
```

## Choosing `cdim`

The `cdim` parameter controls the cell grid resolution. Choose based on your simulation size:

| Simulation Size | Recommended cdim | Total Cells | Use Case |
|----------------|------------------|-------------|----------|
| < 1M particles | 8-16 | 512-4,096 | Small test simulations |
| 1-10M particles | 16-32 | 4,096-32,768 | Medium simulations |
| 10-100M particles | 32-64 | 32,768-262,144 | Large simulations |
| > 100M particles | 64-128 | 262,144-2M | Very large simulations |

**Guidelines:**
- Higher `cdim` → More cells → Better spatial locality → Faster gridder
- But: Very high `cdim` with sparse distributions may waste memory
- For uniform distributions: `cdim ≈ (nparticles^(1/3)) / 10` is a good starting point

## Input File Requirements

Your input HDF5 file must contain:

**Required:**
- Particle coordinates as a (N, 3) array
- Particle masses as a (N,) array

**Optional (but recommended):**
- `Header/BoxSize`: Box dimensions [X, Y, Z]
  - If missing, use `--boxsize` argument
- `Header/NumPart_Total`: Total particle counts
- `Header/Redshift`: Redshift value

**Not required:**
- Cell structure (created by conversion script)
- Velocities
- Particle IDs (will be created if missing)

## Common Issues

### Missing BoxSize

**Error:**
```
ValueError: BoxSize not found in input file and not provided via --boxsize
```

**Solution:**
Provide BoxSize manually:
```bash
--boxsize 100.0 100.0 100.0
```

### Particles Outside Box

Particles with coordinates outside `[0, BoxSize]` will be clamped to the nearest cell boundary.

### Memory Usage

For very large simulations:
- Serial mode: Entire particle array loaded into memory
- MPI mode: Particles split across ranks, reducing per-rank memory

## Verification

After conversion, verify the output structure:

```python
import h5py

with h5py.File('gridder_snapshot.hdf5', 'r') as f:
    # Check required groups
    assert '/PartType1/Coordinates' in f
    assert '/PartType1/Masses' in f
    assert '/Cells/Counts/PartType1' in f
    assert '/Cells/OffsetsInFile/PartType1' in f

    # Check cell structure
    coords = f['/PartType1/Coordinates'][:]
    cell_counts = f['/Cells/Counts/PartType1'][:]

    print(f"Total particles: {len(coords)}")
    print(f"Particles in cells: {cell_counts.sum()}")
    print(f"Non-empty cells: {(cell_counts > 0).sum()}")
```

## Full Workflow Example

Complete example converting a Gadget snapshot:

```bash
# 1. Convert Gadget snapshot to gridder format
python tools/convert_to_gridder_format.py \
    gadget_snapshot_099.hdf5 \
    gridder_input.hdf5 \
    --coordinates-key PartType1/Coordinates \
    --masses-key PartType1/Masses \
    --boxsize 100.0 100.0 100.0 \
    --cdim 32 \
    --copy-header

# 2. Create parameter file (params.yml)
cat > params.yml << EOF
Kernels:
  nkernels: 3
  kernel_radius_1: 1.0
  kernel_radius_2: 2.0
  kernel_radius_3: 5.0

Grid:
  type: uniform
  cdim: 50

Cosmology:
  h: 0.7
  Omega_cdm: 0.25
  Omega_b: 0.05

Tree:
  max_leaf_count: 200

Input:
  filepath: gridder_input.hdf5

Output:
  filepath: output/
  basename: gridded_output.hdf5
  write_masses: 1
EOF

# 3. Run gridder
./build/parent_gridder params.yml 8

# 4. Verify output
ls -lh output/gridded_output.hdf5
```

## See Also

- [Parameter Reference](parameters.md) - Gridder parameter file documentation
- [Quick Start](quickstart.md) - Getting started with the gridder
- [Installation Guide](installation.md) - Building the gridder
