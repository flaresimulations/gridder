#!/bin/bash
#
# Comprehensive test runner for parent_gridder
#
# Usage:
#   ./tests/run_tests.sh [options]
#
# Options:
#   --unit          Run only unit tests (Python pytest)
#   --integration   Run only integration tests (full gridder runs)
#   --all           Run all tests (default)
#   --clean         Clean up test outputs before running
#   --keep          Keep test outputs after running
#   --help          Show this help message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DATA_DIR="$SCRIPT_DIR/data"

# Default options
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_ALL=false
CLEAN_BEFORE=false
KEEP_OUTPUTS=false

# Parse arguments
if [ $# -eq 0 ]; then
    RUN_ALL=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --clean)
            CLEAN_BEFORE=true
            shift
            ;;
        --keep)
            KEEP_OUTPUTS=true
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If --all is set, run everything
if [ "$RUN_ALL" = true ]; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
fi

# Print header
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}    PARENT GRIDDER TEST SUITE${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Clean outputs if requested
if [ "$CLEAN_BEFORE" = true ]; then
    echo -e "${YELLOW}Cleaning test outputs...${NC}"
    rm -f "$DATA_DIR"/*.hdf5
    rm -f "$DATA_DIR"/*.txt
    echo -e "${GREEN}✓ Cleaned${NC}"
    echo ""
fi

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -e "${BLUE}Running: ${test_name}${NC}"

    if eval "$test_command"; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        ((TESTS_PASSED++))
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        ((TESTS_FAILED++))
        echo ""
        return 1
    fi
}

# Build gridder if needed
if [ ! -f "$BUILD_DIR/parent_gridder" ]; then
    echo -e "${YELLOW}Building gridder...${NC}"
    cd "$PROJECT_ROOT"
    cmake -B build
    cmake --build build
    echo -e "${GREEN}✓ Build complete${NC}"
    echo ""
fi

# Run unit tests (pytest)
if [ "$RUN_UNIT" = true ]; then
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}Running Unit Tests (pytest)${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""

    # Temporarily disable exit on error so all tests run
    set +e

    if command -v pytest &> /dev/null; then
        cd "$PROJECT_ROOT"
        run_test "Python unit tests" "pytest tests/test_gridder.py -v"
    else
        echo -e "${YELLOW}Warning: pytest not found, skipping Python unit tests${NC}"
        echo -e "${YELLOW}Install with: pip install pytest${NC}"
        echo ""
    fi

    # Re-enable exit on error
    set -e
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}Running Integration Tests${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""

    # Temporarily disable exit on error so all tests run
    set +e

    # Test 1: Simple test (existing)
    run_test "Simple test (uniform grid)" "bash $SCRIPT_DIR/run_simple_test.sh run"

    # Test 2: File-based grid points
    run_test "File-based grid points" "bash -c \"
        cd $PROJECT_ROOT
        $BUILD_DIR/parent_gridder tests/file_grid_test_params.yml 1 > /dev/null 2>&1 &&
        [ -f tests/data/file_grid_test.hdf5 ]
    \""

    # Test 3: Random grid (if parameters exist)
    if [ -f "$SCRIPT_DIR/random_test_params.yml" ]; then
        run_test "Random grid generation" "bash -c \"
            cd $PROJECT_ROOT
            $BUILD_DIR/parent_gridder tests/random_test_params.yml 1 > /dev/null 2>&1
        \""
    fi

    # Re-enable exit on error
    set -e
fi

# Print summary
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}=============================================${NC}"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo ""

# Clean up if not keeping outputs
if [ "$KEEP_OUTPUTS" = false ] && [ "$CLEAN_BEFORE" = false ]; then
    echo -e "${YELLOW}Cleaning up test outputs...${NC}"
    rm -f "$DATA_DIR"/test*.hdf5
    rm -f "$DATA_DIR"/simple_test_grid.hdf5
    rm -f "$DATA_DIR"/file_grid_test.hdf5
    echo -e "${GREEN}✓ Cleaned${NC}"
fi

# Exit with appropriate code
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    exit 0
else
    echo -e "${RED}=============================================${NC}"
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}=============================================${NC}"
    exit 1
fi
