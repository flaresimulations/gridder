# Custom Test Suite

`tests/test_suite.py` provides end-to-end checks that run built gridder
executables against generated HDF5 snapshots.

## Modes

### Serial

```bash
python3 tests/test_suite.py --mode serial \
  --serial-executable ./build/parent_gridder
```

### MPI

```bash
python3 tests/test_suite.py --mode mpi \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2
```

### Serial/MPI Comparison

```bash
python3 tests/test_suite.py --mode comparison \
  --serial-executable ./build/parent_gridder \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2
```

Use `python3 tests/test_suite.py --help` for the current command-line options.

## Coverage

The suite includes generated uniform, sparse, dense, boundary, empty-kernel,
multi-radius, random-grid, and MPI cases. The test generator writes the HDF5
cell metadata, counts, and offsets required by the gridder.

Additional coverage lives in pytest modules:

- `test_gridder.py` covers core behavior and uniform, random, and file grids.
- `test_conversion.py` covers the HDF5 conversion tool.
- `test_cosmology.py` covers mean-density calculations.

Run those independently with `pytest tests -v`.

## Scope And Limitations

This suite is primarily an integration and regression suite. Some MPI tests
verify successful execution and expected diagnostic messages rather than every
particle transferred through proxy cells. It does not currently provide
performance benchmarks, leak detection, or checkpoint/restart tests.

## Generated Data

Test data is created under `tests/data/`, which is ignored by Git except for
its placeholder. Remove generated data with the cleanup mode supported by the
relevant runner when a clean fixture set is needed.
