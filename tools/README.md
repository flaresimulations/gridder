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

## SWIFT FOF Grid Points

`get_swift_fof_grid_points.py` is specifically for HDF5 FOF outputs written by
SWIFT. It extracts halo centres from a split SWIFT FOF output whose parts are
named `fof_output_NNNN.0.hdf5`, `fof_output_NNNN.1.hdf5`, and so on. It
validates that all expected parts are present, writes one coordinate per line
in the gridder file-grid format, and writes the corresponding FOF mass to a
second text file in the same order. It supports SWIFT outputs whose arrays are
split across every part and outputs that store the complete arrays only in part
`.0`. Masses are preserved in the units stored by SWIFT; no unit conversion is
performed.

```bash
python3 tools/get_swift_fof_grid_points.py \
  /cosma8/data/dp004/flamingo/Runs/L1000N1800/DMO_FIDUCIAL/fof/fof_output_0077/ \
  --output-file fof_grid_points_0077.txt \
  --mass-output-file fof_masses_0077.txt
```

The script detects common centre dataset paths automatically. If the catalogue
uses a different path, supply it explicitly:

```bash
python3 tools/get_swift_fof_grid_points.py /path/to/fof_output_0077/ \
  --centres-dataset Groups/Centres \
  --masses-dataset Groups/Masses \
  --output-file fof_grid_points_0077.txt \
  --mass-output-file fof_masses_0077.txt
```

The centre and mass text files are mutually aligned, but gridder HDF5 output is
reordered by simulation cell. Use the weighting analysis preparation tool to
align the mass array with `Grids/GridPointPositions` before analysing the halo
grid.
