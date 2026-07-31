# Gridder Tools

## Snapshot Converter

`convert_to_gridder_format.py` reads coordinate and mass datasets from an HDF5
file, sorts particles by cell, and writes the cell index required by the C++
gridder.

```bash
python3 tools/convert_to_gridder_format.py input.hdf5 output.hdf5 \
  --coordinates-key DarkMatter/Coordinates \
  --masses-key DarkMatter/Masses \
  --boxsize 100 100 100 \
  --cdim 32
```

Use `--copy-header` only when the source header already has compatible
`BoxSize`, `NumPart_Total`, and `Redshift` attributes. The converter does not
convert units or generate particle IDs.

Gridder-compatible output must use the default `PartType1` particle type.

The converter's MPI mode creates rank files and an HDF5 virtual file, but that
virtual file is not currently safe as gridder input: rank-local sorting does
not provide the global cell ordering assumed by the cell offsets. Use serial
conversion for production gridder input.

See the [conversion guide](../docs/conversion.md) and run:

```bash
python3 tools/convert_to_gridder_format.py --help
```

## Grid Summary

`grid_summary.py` prints a compact summary of gridder HDF5 output:

```bash
python3 tools/grid_summary.py output/grid.hdf5
```

## SOAP Grid Points

`get_fof_grid_points_soap.py` extracts grid-point coordinates from compatible
SOAP catalogues. Inspect its current arguments with:

```bash
python3 tools/get_fof_grid_points_soap.py --help
```
