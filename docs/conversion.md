# Snapshot Conversion

`tools/convert_to_gridder_format.py` creates the cell-indexed HDF5 layout used
by the gridder from coordinate and mass datasets in another HDF5 file.

The converter changes dataset layout only. It does not convert units or infer
whether a source simulation uses the coordinate, mass, and cosmological
conventions expected by the gridder.

## Requirements

- Python 3
- h5py
- numpy
- mpi4py only for the experimental MPI conversion path

## Recommended Serial Conversion

```bash
python3 tools/convert_to_gridder_format.py input.hdf5 gridder_input.hdf5 \
  --coordinates-key DarkMatter/Coordinates \
  --masses-key DarkMatter/Masses \
  --boxsize 100 100 100 \
  --cdim 32
```

Use `--copy-header` when the source `Header` already contains compatible
`BoxSize`, `NumPart_Total`, and `Redshift` attributes:

```bash
python3 tools/convert_to_gridder_format.py input.hdf5 gridder_input.hdf5 \
  --coordinates-key PartType1/Coordinates \
  --masses-key PartType1/Masses \
  --copy-header
```

Run `python3 tools/convert_to_gridder_format.py --help` for all current
arguments.

## Particle Type

Gridder-compatible output must use `PartType1`, which is the converter default.
Although the converter accepts `--particle-type`, the C++ reader currently
hardcodes `PartType1` for particles, counts, and offsets.

## Output Layout

Serial conversion sorts particles by cell and writes:

```text
Header
  attributes: BoxSize, NumPart_Total, Redshift
PartType1/Coordinates
PartType1/Masses
Cells/Meta-data
  attributes: dimension, size
Cells/Counts/PartType1
Cells/OffsetsInFile/PartType1
```

Particle IDs and velocities are not generated because the gridder does not
read them.

## Choosing `cdim`

`--cdim` sets the number of cells along each dimension; the default is `16`.
More cells can improve spatial locality but increase cell metadata and can
produce many empty cells. Benchmark representative data rather than relying on
a fixed particle-count rule.

## MPI Conversion Limitation

The converter has an MPI mode that writes rank files and an HDF5 virtual file.
Do not use that virtual file as gridder input at present. Each rank sorts only
its own particle chunk, while the virtual file concatenates rank chunks; this
does not establish the global cell ordering assumed by
`Cells/OffsetsInFile/PartType1`.

Use serial conversion for gridder-compatible output until the MPI converter
implements global cell ordering or the reader becomes rank-layout aware.

## Verify Output

```python
import h5py

with h5py.File("gridder_input.hdf5", "r") as handle:
    required = [
        "PartType1/Coordinates",
        "PartType1/Masses",
        "Cells/Counts/PartType1",
        "Cells/OffsetsInFile/PartType1",
    ]
    for path in required:
        assert path in handle, path

    counts = handle["Cells/Counts/PartType1"][:]
    coordinates = handle["PartType1/Coordinates"]
    assert counts.sum() == len(coordinates)
```

Also verify that header units and the `Cosmology` values in the gridder
parameter file describe the converted particle data.
