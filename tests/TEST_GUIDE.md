# Test Guide for Parent Gridder

This document provides comprehensive information about testing the parent_gridder application.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Writing New Tests](#writing-new-tests)
5. [CI/CD Integration](#cicd-integration)
6. [Troubleshooting](#troubleshooting)

## Overview

The parent_gridder test suite includes:

- **Unit tests**: Python-based tests using pytest for individual components
- **Integration tests**: End-to-end tests of the full gridder workflow
- **File-based grid point tests**: Specific tests for the new file grid reading feature
- **CI/CD automation**: GitHub Actions workflows for automated testing

## Test Structure

```
tests/
├── test_gridder.py              # Main pytest test suite
├── run_tests.sh                 # Master test runner script
├── run_simple_test.sh          # Simple integration test
├── simple_test_params.yml      # Parameters for simple test
├── file_grid_test_params.yml   # Parameters for file grid test
├── data/                        # Test data directory
│   ├── grid_points_files/       # Grid point test files
│   │   ├── valid_points.txt     # Valid grid points
│   │   ├── with_comments.txt    # Test comment handling
│   │   └── malformed.txt        # Test error handling
│   ├── simple_test.hdf5        # Test snapshot (auto-generated)
│   └── *_test.hdf5             # Test outputs
├── TEST_GUIDE.md               # This file
└── README.md                    # Quick start guide
```

## Running Tests

### Quick Start

```bash
# Run all tests
./tests/run_tests.sh --all

# Run only unit tests
./tests/run_tests.sh --unit

# Run only integration tests
./tests/run_tests.sh --integration

# Run tests and keep outputs for inspection
./tests/run_tests.sh --all --keep
```

### Python Unit Tests

```bash
# Run all pytest tests
pytest tests/test_gridder.py -v

# Run specific test class
pytest tests/test_gridder.py::TestFileGridPoints -v

# Run specific test
pytest tests/test_gridder.py::TestFileGridPoints::test_valid_grid_points_file -v

# Run with markers
pytest tests/test_gridder.py -m file_grid -v
```

### Integration Tests

```bash
# Simple test (1 particle, predictable results)
./tests/run_simple_test.sh

# File-based grid points
cd build
./parent_gridder ../tests/file_grid_test_params.yml 1
```

## Test Categories

### 1. File-Based Grid Point Tests

Tests for the new file grid reading feature (TestFileGridPoints class):

- **test_valid_grid_points_file**: Reads a valid grid points file and verifies output
- **test_grid_points_with_comments**: Tests parsing with comments and whitespace
- **test_malformed_grid_points**: Tests graceful handling of malformed lines
- **test_missing_grid_file**: Tests error handling for missing files

**Grid Point File Format:**
```
# Comments start with #
x1 y1 z1
x2 y2 z2
# Blank lines and whitespace are ignored
x3 y3 z3
```

### 2. Uniform Grid Tests

Tests for uniform grid generation (TestUniformGrid class):

- **test_uniform_grid_simple**: Basic uniform grid functionality

### 3. Output Validation Tests

Tests for output file structure and correctness (TestOutputValidation class):

- **test_output_hdf5_structure**: Validates HDF5 output structure
- Checks for required groups and datasets
- Validates kernel data
- Verifies write_masses parameter is honored

## Writing New Tests

### Adding a Unit Test

1. Open `tests/test_gridder.py`
2. Add a new test method to the appropriate class:

```python
class TestFileGridPoints:
    def test_my_new_feature(self, build_gridder, test_snapshot):
        """Test description."""
        # Setup
        param_file = TESTS_DIR / "my_test_params.yml"
        output_file = DATA_DIR / "my_test_output.hdf5"

        # Execute
        result = subprocess.run(
            [str(build_gridder), str(param_file), "1"],
            capture_output=True,
            text=True
        )

        # Verify
        assert result.returncode == 0, f"Gridder failed: {result.stderr}"
        assert output_file.exists(), "Output file not created"

        # Validate output
        with h5py.File(output_file, 'r') as f:
            # Add your validations here
            pass
```

### Adding an Integration Test

1. Create a parameter file in `tests/`
2. Create test data in `tests/data/`
3. Add test case to `tests/run_tests.sh`:

```bash
run_test "My integration test" "bash -c '
    cd $PROJECT_ROOT
    $BUILD_DIR/parent_gridder tests/my_test_params.yml 1 > /dev/null 2>&1 &&
    [ -f tests/data/my_expected_output.hdf5 ]
'"
```

### Test Fixtures

Available pytest fixtures:

- **build_gridder**: Ensures gridder is built, returns path to executable
- **test_snapshot**: Creates/returns path to test snapshot file

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Three test jobs run in parallel:
1. **test-serial**: Serial build tests
2. **test-mpi**: MPI build tests
3. **test-debug**: Debug build tests

### Local Pre-commit Testing

Before committing, run:

```bash
# Quick test
./tests/run_tests.sh --unit

# Full test
./tests/run_tests.sh --all
```

## Test Data Files

### Grid Point Files

**valid_points.txt**: 5 valid grid points within a 10x10x10 box
- Tests basic file reading
- All points should be processed

**with_comments.txt**: Grid points with extensive comments and whitespace
- Tests robust parsing
- Should ignore comments (lines starting with #)
- Should handle empty lines and whitespace-only lines

**malformed.txt**: Intentionally malformed lines
- Tests error recovery
- Should skip invalid lines gracefully
- Should process valid lines despite errors

### Parameter Files

**simple_test_params.yml**: Basic test with 1 particle
- Uniform grid (3x3x3)
- 3 kernel radii (0.5, 1.0, 2.0)
- Predictable results for validation

**file_grid_test_params.yml**: File-based grid points
- Reads from valid_points.txt
- Same kernels as simple test
- Tests file reading functionality

## Troubleshooting

### Test Failures

**"Gridder executable not found"**
- Solution: Run `cmake --build build` to build the gridder

**"pytest: command not found"**
- Solution: Install pytest with `pip install pytest`

**"Failed to create test snapshot"**
- Check that `make_test_snap.py` works correctly
- Verify donor snapshot exists or can be created
- Solution: `python make_test_snap.py --help`

**"HDF5 library not found"**
- Solution: Install HDF5 development files
  - Ubuntu/Debian: `sudo apt-get install libhdf5-dev`
  - macOS: `brew install hdf5`

### Debugging Tests

Run pytest with extra verbosity:
```bash
pytest tests/test_gridder.py -vv --tb=long
```

Keep test outputs for inspection:
```bash
./tests/run_tests.sh --all --keep
ls tests/data/
```

Run gridder manually with verbose output:
```bash
./build/parent_gridder tests/file_grid_test_params.yml 1
```

## Expected Test Output

### Successful Test Run

```
=============================================
    PARENT GRIDDER TEST SUITE
=============================================

Running Unit Tests (pytest)
============================================= test session starts ==============================================
tests/test_gridder.py::TestFileGridPoints::test_valid_grid_points_file PASSED                    [ 20%]
tests/test_gridder.py::TestFileGridPoints::test_grid_points_with_comments PASSED                [ 40%]
tests/test_gridder.py::TestFileGridPoints::test_malformed_grid_points PASSED                    [ 60%]
tests/test_gridder.py::TestUniformGrid::test_uniform_grid_simple PASSED                         [ 80%]
tests/test_gridder.py::TestOutputValidation::test_output_hdf5_structure PASSED                  [100%]

================================================ 5 passed in 10.2s =============================================

Running Integration Tests
=============================================
✓ Simple test (uniform grid) PASSED
✓ File-based grid points PASSED

=============================================
Test Summary
=============================================
Tests passed: 7
Tests failed: 0

✓ ALL TESTS PASSED!
=============================================
```

## Adding Tests for New Features

When adding a new feature to the gridder:

1. **Write tests first** (TDD approach):
   - Add test cases for the expected behavior
   - Add test cases for edge cases and error conditions

2. **Implement the feature**:
   - Develop the C++ code
   - Run tests frequently during development

3. **Validate with integration tests**:
   - Create parameter files for realistic scenarios
   - Verify outputs manually once
   - Automate the validation in test suite

4. **Update documentation**:
   - Add test descriptions to this guide
   - Update README.md if needed
   - Document expected behavior

## Contact

For questions about testing:
- Check existing tests for examples
- Review test output carefully
- Open an issue with test output if tests fail unexpectedly
