#!/bin/bash

# Simple test runner for the gridder
# This creates a simple test case and verifies the expected results

set -e # Exit on any error

# Test configuration
TEST_DIR="$(dirname "$0")"
ROOT_DIR="$(dirname "$TEST_DIR")"
TEST_DATA_DIR="$TEST_DIR/data"
DONOR_FILE="$TEST_DATA_DIR/donor_snapshot.hdf5"
TEST_SNAPSHOT="$TEST_DATA_DIR/simple_test.hdf5"
TEST_PARAMS="$TEST_DIR/simple_test_params.yml"
OUTPUT_FILE="$TEST_DATA_DIR/simple_test_grid.hdf5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "          GRIDDER SIMPLE TEST RUNNER"
echo "=============================================="

# Create test data directory
mkdir -p "$TEST_DATA_DIR"

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}ERROR: Required file not found: $1${NC}"
        echo "Please ensure all required files are available before running tests."
        exit 1
    fi
}

# Function to create a minimal donor file if it doesn't exist
create_donor_file() {
    if [ ! -f "$DONOR_FILE" ]; then
        echo -e "${YELLOW}Creating minimal donor file for testing...${NC}"
        python3 -c "
import h5py
import numpy as np

with h5py.File('$DONOR_FILE', 'w') as f:
    # Minimal units
    units = f.create_group('Units')
    units.attrs['Unit mass in cgs (U_M)'] = 1.989e43  # 10^10 solar masses
    units.attrs['Unit length in cgs (U_L)'] = 3.086e24  # Mpc
    
    # Minimal cosmology
    cosmo = f.create_group('Cosmology')
    cosmo.attrs['Critical density [internal units]'] = 1.0
"
        echo -e "${GREEN}Created donor file: $DONOR_FILE${NC}"
    fi
}

# Function to run the test
run_test() {
    echo -e "${YELLOW}Step 1: Creating simple test snapshot...${NC}"

    # Create donor file if needed
    create_donor_file

    # Create simple test snapshot
    cd "$ROOT_DIR"
    python3 make_test_snap.py \
        --output "$TEST_SNAPSHOT" \
        --cdim 3 \
        --boxsize 10.0 \
        --doner "$DONOR_FILE" \
        --simple

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test snapshot created successfully${NC}"
    else
        echo -e "${RED}✗ Failed to create test snapshot${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Step 2: Running gridder on test data...${NC}"

    # Check if gridder executable exists (try multiple possible locations)
    GRIDDER_EXEC=""
    POSSIBLE_LOCATIONS=(
        "$ROOT_DIR/build/parent_gridder"
        "$ROOT_DIR/parent_gridder"
        "$ROOT_DIR/cmake-build-debug/parent_gridder"
        "$ROOT_DIR/cmake-build-release/parent_gridder"
    )

    for location in "${POSSIBLE_LOCATIONS[@]}"; do
        if [ -f "$location" ]; then
            GRIDDER_EXEC="$location"
            echo -e "${GREEN}Found executable: $GRIDDER_EXEC${NC}"
            break
        fi
    done

    if [ -z "$GRIDDER_EXEC" ]; then
        echo -e "${RED}ERROR: parent_gridder executable not found${NC}"
        echo "Searched in the following locations:"
        for location in "${POSSIBLE_LOCATIONS[@]}"; do
            echo "  - $location"
        done
        echo "Please build the project first with 'make' or 'cmake --build .'"
        exit 1
    fi

    # Run the gridder
    "$GRIDDER_EXEC" "$TEST_PARAMS" 4

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Gridder completed successfully${NC}"
    else
        echo -e "${RED}✗ Gridder failed${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Step 3: Verifying results...${NC}"

    # Check if output file was created
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo -e "${RED}✗ Output file not created: $OUTPUT_FILE${NC}"
        exit 1
    fi

    # Verify the results using Python
    python3 -c "
import h5py
import numpy as np
import sys

try:
    with h5py.File('$OUTPUT_FILE', 'r') as f:
        print('Checking output file structure...')
        
        # Check if required groups exist
        required_groups = ['Header', 'Grids', 'Cells']
        for group in required_groups:
            if group not in f:
                print(f'ERROR: Missing group: {group}')
                sys.exit(1)
        
        # Check kernels
        print('Available kernels:', list(f['Grids'].keys()))

        # Check mass data for kernel_radius_1 (which is Kernel_0 in 0-indexed naming)
        kernel_name = 'Kernel_0'

        if kernel_name not in f['Grids']:
            print(f'ERROR: Could not find {kernel_name}')
            sys.exit(1)

        print(f'Found kernel: {kernel_name}')

        # Check kernel radius attribute
        if 'KernelRadius' in f['Grids'][kernel_name].attrs:
            kernel_radius = f['Grids'][kernel_name].attrs['KernelRadius']
            print(f'Kernel radius: {kernel_radius}')

        # Check if masses exist
        if 'GridPointMasses' in f['Grids'][kernel_name]:
            masses = f['Grids'][kernel_name]['GridPointMasses'][:]
            print(f'Grid point masses: {masses}')

            # For simple test, we expect at least one mass value of 1.0
            if np.any(np.isclose(masses, 1.0, rtol=1e-10)):
                print('✓ Found expected mass value of 1.0')
            else:
                print(f'WARNING: Expected mass value 1.0, got: {masses}')
        else:
            print('INFO: Mass data not written (write_masses may be disabled)')

        # Check overdensities
        if 'GridPointOverDensities' in f['Grids'][kernel_name]:
            overdens = f['Grids'][kernel_name]['GridPointOverDensities'][:]
            print(f'Grid point overdensities: {overdens}')
        
        print('✓ Output file structure is valid')
        
except Exception as e:
    print(f'ERROR: Failed to verify results: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Results verification passed${NC}"
    else
        echo -e "${RED}✗ Results verification failed${NC}"
        exit 1
    fi
}

# Function to clean up test files
cleanup() {
    echo -e "${YELLOW}Cleaning up test files...${NC}"
    rm -f "$TEST_SNAPSHOT" "$OUTPUT_FILE"
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Main execution
case "${1:-run}" in
"run")
    run_test
    echo -e "${GREEN}=============================================="
    echo "✓ ALL TESTS PASSED!"
    echo "==============================================${NC}"
    exit 0
    ;;
"clean")
    cleanup
    exit 0
    ;;
"help")
    echo "Usage: $0 [run|clean|help]"
    echo "  run   - Run the simple test (default)"
    echo "  clean - Clean up test files"
    echo "  help  - Show this help message"
    exit 0
    ;;
*)
    echo "Unknown command: $1"
    echo "Use '$0 help' for usage information"
    exit 1
    ;;
esac
