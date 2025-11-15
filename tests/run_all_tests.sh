#!/bin/bash
# Comprehensive test runner for gridder
# Runs tests in both serial and MPI modes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  GRIDDER COMPREHENSIVE TEST SUITE RUNNER"
echo "=============================================="
echo ""

# Check if executables exist
if [ ! -f "./build/parent_gridder" ]; then
    echo -e "${RED}Error: Serial executable not found. Run: cmake -B build && cmake --build build${NC}"
    exit 1
fi

if [ ! -f "./build_mpi/parent_gridder" ]; then
    echo -e "${YELLOW}Warning: MPI executable not found. Skipping MPI tests.${NC}"
    echo -e "${YELLOW}To enable MPI tests, run: cmake -B build_mpi -DENABLE_MPI=ON && cmake --build build_mpi${NC}"
    RUN_MPI=false
else
    RUN_MPI=true
fi

echo ""

# Run serial tests
echo -e "${YELLOW}Running SERIAL tests...${NC}"
echo "----------------------------------------"
python3 tests/test_suite.py --mode serial --executable ./build/parent_gridder
SERIAL_EXIT=$?

echo ""

# Run MPI tests if available
if [ "$RUN_MPI" = true ]; then
    echo -e "${YELLOW}Running MPI tests (2 ranks)...${NC}"
    echo "----------------------------------------"
    python3 tests/test_suite.py --mode mpi --mpi-executable ./build_mpi/parent_gridder --ranks 2
    MPI_EXIT=$?

    echo ""
    echo -e "${YELLOW}Running MPI tests (4 ranks)...${NC}"
    echo "----------------------------------------"
    python3 tests/test_suite.py --mode mpi --mpi-executable ./build_mpi/parent_gridder --ranks 4
    MPI4_EXIT=$?
else
    MPI_EXIT=0
    MPI4_EXIT=0
fi

# Summary
echo ""
echo "=============================================="
echo "  FINAL SUMMARY"
echo "=============================================="

TOTAL_FAILED=0

if [ $SERIAL_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Serial tests: PASSED"
else
    echo -e "${RED}✗${NC} Serial tests: FAILED"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

if [ "$RUN_MPI" = true ]; then
    if [ $MPI_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓${NC} MPI tests (2 ranks): PASSED"
    else
        echo -e "${RED}✗${NC} MPI tests (2 ranks): FAILED"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi

    if [ $MPI4_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓${NC} MPI tests (4 ranks): PASSED"
    else
        echo -e "${RED}✗${NC} MPI tests (4 ranks): FAILED"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
fi

echo "=============================================="

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}$TOTAL_FAILED TEST SUITE(S) FAILED${NC}"
    exit 1
fi
