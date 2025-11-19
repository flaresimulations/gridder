#!/bin/bash
#
# Build in debug mode and run comprehensive tests
# This script compiles with DEBUGGING_CHECKS enabled and runs the full test suite
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "  BUILD AND TEST IN DEBUG MODE"
echo -e "==============================================${NC}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Clean previous debug build
echo -e "${YELLOW}Cleaning previous debug build...${NC}"
rm -rf build_debug

# Configure with debug mode
echo -e "${YELLOW}Configuring debug build...${NC}"
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug

# Build
echo -e "${YELLOW}Building in debug mode...${NC}"
cmake --build build_debug

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Check executable
if [ ! -f "build_debug/parent_gridder" ]; then
    echo -e "${RED}✗ Executable not found: build_debug/parent_gridder${NC}"
    exit 1
fi

echo -e "${BLUE}Running comprehensive test suite...${NC}"
echo ""

# Run file-based gridding tests
echo -e "${YELLOW}=== File-Based Gridding Tests ===${NC}"
python3 tests/test_file_gridding.py build_debug/parent_gridder

FILE_TEST_EXIT=$?

echo ""

# Run simple tests
echo -e "${YELLOW}=== Simple Test ===${NC}"
bash tests/run_simple_test.sh

SIMPLE_TEST_EXIT=$?

echo ""

# Summary
echo -e "${BLUE}=============================================="
echo "  TEST SUMMARY"
echo -e "==============================================${NC}"

TOTAL_FAILED=0

if [ $FILE_TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} File-based gridding tests: PASSED"
else
    echo -e "${RED}✗${NC} File-based gridding tests: FAILED"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

if [ $SIMPLE_TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Simple tests: PASSED"
else
    echo -e "${RED}✗${NC} Simple tests: FAILED"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

echo -e "${BLUE}==============================================${NC}"

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}$TOTAL_FAILED TEST SUITE(S) FAILED${NC}"
    echo ""
    echo "Debug build is available at: build_debug/parent_gridder"
    echo "You can run individual tests or use a debugger to investigate failures."
    exit 1
fi
