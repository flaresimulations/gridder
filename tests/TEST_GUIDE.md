# Test Guide

This guide describes the available test entry points. See
[`README.md`](README.md) for a short command reference.

## Pytest Suites

Run all pytest-discovered tests:

```bash
pytest tests -v
```

The main modules are:

- `test_gridder.py`: grid construction, file/random grids, output, and errors
- `test_conversion.py`: serial snapshot conversion
- `test_cosmology.py`: cosmological density calculations

Run a module or test directly when iterating:

```bash
pytest tests/test_gridder.py -v
pytest tests/test_gridder.py::TestFileGridPoints -v
```

## Shell Runners

```bash
./tests/run_tests.sh --unit
./tests/run_tests.sh --integration
./tests/run_tests.sh --all
./tests/run_simple_test.sh
```

`run_tests.sh` is a focused build/integration runner. It does not invoke every
pytest module or every custom-suite mode.

## Custom Serial And MPI Suite

```bash
python3 tests/test_suite.py --mode serial \
  --serial-executable ./build/parent_gridder

python3 tests/test_suite.py --mode mpi \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2

python3 tests/test_suite.py --mode comparison \
  --serial-executable ./build/parent_gridder \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2
```

Some custom MPI cases are smoke tests or inspect diagnostics rather than
performing exhaustive numerical validation of communication internals.

## Writing Tests

- Put pytest regression tests beside the related existing module.
- Use fixtures and temporary paths rather than committing generated HDF5 data.
- Put generated artifacts under `tests/data/`; this directory is ignored.
- Verify numerical output, not only successful process exit, where practical.
- Add MPI-specific coverage when behavior differs across ranks.

Generate a compatible snapshot manually with:

```bash
python3 tests/make_test_snap.py --help
```

## Debugging

```bash
pytest tests/test_gridder.py -vv --tb=long
./tests/run_tests.sh --all --keep
./build/parent_gridder tests/file_grid_test_params.yml 1 0 2
```

If configuration fails, confirm the HDF5 development libraries are installed
and rebuild the relevant serial, MPI, or debug build directory.
