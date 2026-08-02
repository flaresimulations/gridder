#!/bin/bash

# Build documentation for FLARES-2 Gridder
# Installs dependencies if needed and builds static site with MkDocs

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  FLARES-2 Gridder Documentation Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is required but not found${NC}"
    echo "Please install Python 3 and try again"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Found Python $PYTHON_VERSION"

# Use the selected Python interpreter for installation and execution. This
# avoids mixing environments when pip3 or mkdocs resolves through pyenv.
if ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}ERROR: pip is not available for python3${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Found pip for python3"

# Check all dependencies required by mkdocs.yml.
if ! python3 -c "import mkdocs, material, mkdocs_autorefs" &> /dev/null; then
    echo -e "${YELLOW}Documentation dependencies not found. Installing...${NC}"
    echo ""

    echo -e "${BLUE}Installing documentation requirements...${NC}"
    python3 -m pip install --user -r requirements-docs.txt

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install documentation requirements${NC}"
        echo "Try running manually: python3 -m pip install --user -r requirements-docs.txt"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Dependencies installed successfully"
    echo ""
else
    MKDOCS_VERSION=$(python3 -m mkdocs --version)
    echo -e "${GREEN}✓${NC} Found $MKDOCS_VERSION"
fi

echo ""
echo -e "${BLUE}Building documentation...${NC}"
echo ""

# Build the documentation
python3 -m mkdocs build --clean

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Documentation build failed${NC}"
    echo "Check the error messages above for details"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Documentation built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "📁 Output directory: ${BLUE}site/${NC}"
echo -e "🏠 Homepage:         ${BLUE}file://$SCRIPT_DIR/site/index.html${NC}"
echo ""
echo -e "To view the documentation:"
echo -e "  ${YELLOW}open site/index.html${NC}          (macOS)"
echo -e "  ${YELLOW}xdg-open site/index.html${NC}      (Linux)"
echo -e "  ${YELLOW}start site/index.html${NC}         (Windows)"
echo ""
echo -e "To serve locally with live reload:"
echo -e "  ${YELLOW}python3 -m mkdocs serve${NC}"
echo -e "  Then open: ${BLUE}http://127.0.0.1:8000${NC}"
echo ""
echo -e "To deploy to GitHub Pages:"
echo -e "  ${YELLOW}python3 -m mkdocs gh-deploy${NC}"
echo ""
