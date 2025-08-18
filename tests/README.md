# Gridder Tests

This directory contains test scripts and data for verifying the gridder functionality.

## Simple Test

The simple test creates a minimal test case with predictable results:

- **1 particle** with mass = 1.0 at the center of the simulation box
- **3 kernel radii**: 0.5, 1.0, and 2.0
- **Expected results** for the grid point at center:
  - Kernel radius 0.5: mass = 0.0 (particle outside sphere)
  - Kernel radius 1.0: mass = 1.0 (particle inside sphere) ✅
  - Kernel radius 2.0: mass = 1.0 (particle inside sphere)

## Running Tests

### Quick Test
```bash
cd tests
./run_simple_test.sh
```

### Manual Test Steps
```bash
# 1. Create test data
python3 make_test_snap.py --output tests/data/simple_test.hdf5 --cdim 3 --boxsize 10.0 --doner some_snapshot.hdf5 --simple

# 2. Run gridder
./parent_gridder tests/simple_test_params.yml 0

# 3. Check results in tests/data/simple_test_grid.hdf5
```

### Test Commands
- `./run_simple_test.sh run` - Run the test (default)
- `./run_simple_test.sh clean` - Clean up test files
- `./run_simple_test.sh help` - Show help

## Test Structure

```
tests/
├── run_simple_test.sh          # Main test runner script
├── simple_test_params.yml      # Parameters for simple test
├── data/                       # Test data directory (created automatically)
│   ├── donor_snapshot.hdf5     # Minimal donor file (created if needed)
│   ├── simple_test.hdf5        # Test input snapshot
│   └── simple_test_grid.hdf5   # Test output grid
└── README.md                   # This file
```

## Test Features

- **Automatic setup**: Creates test data and donor files as needed
- **Error checking**: Validates each step and provides clear error messages
- **Result verification**: Checks output file structure and expected values
- **Colored output**: Easy to see test status and results
- **Cleanup**: Option to clean up test files after running

## Expected Output

When the test passes, you should see:
```
==============================================
          GRIDDER SIMPLE TEST RUNNER
==============================================
Step 1: Creating simple test snapshot...
✓ Test snapshot created successfully
Step 2: Running gridder on test data...
✓ Gridder completed successfully  
Step 3: Verifying results...
✓ Found expected mass value of 1.0
✓ Results verification passed
==============================================
✓ ALL TESTS PASSED!
==============================================
```

This confirms that your gridder correctly calculates mass = 1.0 for a kernel radius of 1.0 in the simple test case.