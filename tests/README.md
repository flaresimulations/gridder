# Tests

The repository contains pytest suites, a custom serial/MPI suite, and focused
shell runners.

## Pytest

Pytest discovers tests in:

- `tests/test_gridder.py`: core behavior, uniform/random/file grids, and input validation
- `tests/test_conversion.py`: snapshot conversion
- `tests/test_cosmology.py`: cosmological density calculations

Run all discovered pytest tests from the repository root:

```bash
pytest tests -v
```

Some cosmology comparisons are skipped when optional Python dependencies are
not installed.

## Focused Runner

```bash
./tests/run_tests.sh --all
```

`run_tests.sh` is a focused build and integration runner, not an exhaustive
wrapper around every pytest module or custom-suite mode. Use `pytest tests -v`
and the commands below for broader coverage.

## Custom Suite

```bash
# Serial
python3 tests/test_suite.py --mode serial \
  --executable ./build/parent_gridder

# MPI
python3 tests/test_suite.py --mode mpi \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2

# Compare serial and MPI output
python3 tests/test_suite.py --mode comparison \
  --executable ./build/parent_gridder \
  --mpi-executable ./build_mpi/parent_gridder --ranks 2
```

The custom suite exercises particle loading, kernel calculations, and MPI
paths. Some MPI checks are smoke tests or inspect diagnostic messages rather
than exhaustively validating every exchanged value.

## Simple Sanity Test

```bash
./tests/run_simple_test.sh
```

The generated centered particle is inside a centered radius-0.5 kernel. The
runner is intended as a quick diagnostic and prints some numerical warnings
without converting every warning into a failing exit status.

Generate test input manually with:

```bash
python3 tests/make_test_snap.py --help
```

Generated files are written under `tests/data/` and are ignored by Git.
