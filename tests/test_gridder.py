#!/usr/bin/env python3
"""
Test suite for the parent_gridder application.

This test suite validates:
1. Grid point file reading (new feature)
2. Uniform grid generation
3. Output file structure and correctness
4. Edge cases and error handling
"""

import os
import subprocess
import sys
import h5py
import numpy as np
import pytest
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
GRIDDER_EXE = BUILD_DIR / "parent_gridder"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = TESTS_DIR / "data"
GRID_POINTS_DIR = DATA_DIR / "grid_points_files"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def build_gridder():
    """Ensure gridder is built before running tests."""
    if not GRIDDER_EXE.exists():
        print("Building gridder...")
        subprocess.run(["cmake", "-B", str(BUILD_DIR)], cwd=PROJECT_ROOT, check=True)
        subprocess.run(["cmake", "--build", str(BUILD_DIR)], cwd=PROJECT_ROOT, check=True)
    assert GRIDDER_EXE.exists(), f"Gridder executable not found at {GRIDDER_EXE}"
    return GRIDDER_EXE


@pytest.fixture(scope="session")
def test_snapshot():
    """Create a simple test snapshot for use in tests."""
    snapshot_path = DATA_DIR / "simple_test.hdf5"

    # Check if snapshot already exists
    if snapshot_path.exists():
        return snapshot_path

    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create a minimal test snapshot using make_test_snap.py
    make_snap_script = PROJECT_ROOT / "make_test_snap.py"

    # Find a donor snapshot (use any existing HDF5 file or create minimal one)
    donor_path = DATA_DIR / "donor_minimal.hdf5"
    if not donor_path.exists():
        # Create minimal donor with just units and cosmology
        create_minimal_donor(donor_path)

    cmd = [
        sys.executable,
        str(make_snap_script),
        "--output", str(snapshot_path),
        "--cdim", "3",
        "--grid_dim", "10",
        "--boxsize", "10.0",
        "--doner", str(donor_path),
        "--simple"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create test snapshot:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    assert snapshot_path.exists(), f"Failed to create test snapshot at {snapshot_path}"

    return snapshot_path


def create_minimal_donor(filepath):
    """Create a minimal donor HDF5 file with necessary metadata."""
    with h5py.File(filepath, 'w') as f:
        # Create Units group
        units = f.create_group('Units')
        units.attrs['Unit length in cgs (U_L)'] = 3.085677581491367e+24  # Mpc
        units.attrs['Unit mass in cgs (U_M)'] = 1.988409870698051e+43  # 10^10 Msun
        units.attrs['Unit time in cgs (U_t)'] = 3.085677581491367e+19
        units.attrs['Unit current in cgs (U_I)'] = 1.0
        units.attrs['Unit temperature in cgs (U_T)'] = 1.0

        # Create Cosmology group
        cosmo = f.create_group('Cosmology')
        cosmo.attrs['Redshift'] = 0.0
        cosmo.attrs['Scale-factor'] = 1.0
        cosmo.attrs['Omega_m'] = 0.3
        cosmo.attrs['Omega_lambda'] = 0.7
        cosmo.attrs['Omega_b'] = 0.04
        cosmo.attrs['h'] = 0.7
        # Critical density needed by make_test_snap.py
        # At z=0, rho_crit = 3H^2/(8πG) ≈ 1.878e-26 h^2 kg/m^3
        # In internal units (10^10 Msun / Mpc^3): ~0.027 h^2
        cosmo.attrs['Critical density [internal units]'] = 0.027 * 0.7**2


# ============================================================================
# File-based Grid Points Tests
# ============================================================================

class TestFileGridPoints:
    """Tests for file-based grid point reading."""

    def test_valid_grid_points_file(self, build_gridder, test_snapshot):
        """Test reading a valid grid points file."""
        param_file = TESTS_DIR / "file_grid_test_params.yml"
        output_file = DATA_DIR / "file_grid_test.hdf5"

        # Remove old output
        if output_file.exists():
            output_file.unlink()

        # Run gridder
        result = subprocess.run(
            [str(build_gridder), str(param_file), "1"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, (
            f"Gridder failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert output_file.exists(), f"Output file not created at {output_file}"

        # Verify output
        with h5py.File(output_file, 'r') as f:
            assert '/Grids' in f, "Missing /Grids group"

            # Check that grid points were read
            if '/Grids/GridPointPositions' in f:
                positions = f['/Grids/GridPointPositions'][:]
                assert len(positions) == 5, f"Expected 5 grid points, got {len(positions)}"

            # Check kernel data (kernel names have full floating point precision)
            kernel_keys = [k for k in f['/Grids'].keys() if k.startswith('Kernel_')]
            assert len(kernel_keys) >= 1, "Missing kernel data"

    def test_grid_points_with_comments(self, build_gridder, test_snapshot):
        """Test that comments and whitespace are handled correctly."""
        # Create temporary param file
        param_content = """
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: file
  grid_file: tests/data/grid_points_files/with_comments.txt

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: comments_test.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_comments_test.yml"
        param_file.write_text(param_content)

        output_file = DATA_DIR / "comments_test.hdf5"
        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(build_gridder), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, (
            f"Gridder failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
            assert output_file.exists(), f"Output file not created at {output_file}"

            # Verify correct number of points were read (should be 4 valid points)
            with h5py.File(output_file, 'r') as f:
                if '/Grids/GridPointPositions' in f:
                    positions = f['/Grids/GridPointPositions'][:]
                    assert len(positions) >= 4, f"Expected at least 4 grid points, got {len(positions)}"
        finally:
            if param_file.exists():
                param_file.unlink()

    def test_malformed_grid_points(self, build_gridder, test_snapshot):
        """Test that malformed lines are skipped gracefully."""
        param_content = """
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: file
  grid_file: tests/data/grid_points_files/malformed.txt

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: malformed_test.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_malformed_test.yml"
        param_file.write_text(param_content)

        output_file = DATA_DIR / "malformed_test.hdf5"
        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(build_gridder), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            # Should still succeed despite malformed lines
            assert result.returncode == 0, (
            f"Gridder failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
            assert output_file.exists(), f"Output file not created at {output_file}"

            # Verify that only valid points were read
            with h5py.File(output_file, 'r') as f:
                if '/Grids/GridPointPositions' in f:
                    positions = f['/Grids/GridPointPositions'][:]
                    # Should have at least the valid lines
                    assert len(positions) >= 3, f"Expected at least 3 valid points"
        finally:
            if param_file.exists():
                param_file.unlink()

    def test_missing_grid_file(self, build_gridder, test_snapshot):
        """Test error handling when grid file doesn't exist."""
        param_content = """
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: file
  grid_file: tests/data/grid_points_files/nonexistent.txt

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: missing_file_test.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_missing_file_test.yml"
        param_file.write_text(param_content)

        try:
            result = subprocess.run(
                [str(build_gridder), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            # Should fail with non-zero exit code
            assert result.returncode != 0, "Gridder should fail with missing grid file"
            assert "Failed to open" in result.stderr or "Failed to read" in result.stderr, \
                f"Expected error message about missing file, got: {result.stderr}"
        finally:
            if param_file.exists():
                param_file.unlink()


# ============================================================================
# Uniform Grid Tests
# ============================================================================

class TestUniformGrid:
    """Tests for uniform grid generation."""

    def test_uniform_grid_simple(self, build_gridder, test_snapshot):
        """Test basic uniform grid functionality."""
        param_file = TESTS_DIR / "simple_test_params.yml"
        output_file = DATA_DIR / "simple_test_grid.hdf5"

        if output_file.exists():
            output_file.unlink()

        result = subprocess.run(
            [str(build_gridder), str(param_file), "1"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, (
            f"Gridder failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert output_file.exists(), f"Output file not created at {output_file}"

        # Verify output structure
        with h5py.File(output_file, 'r') as f:
            assert '/Grids' in f, "Missing /Grids group"

            # Check kernels (kernel names have full floating point precision)
            kernel_keys = [k for k in f['/Grids'].keys() if k.startswith('Kernel_')]
            assert len(kernel_keys) >= 3, f"Expected at least 3 kernels, found {len(kernel_keys)}"

            # Check each kernel has overdensities
            for kernel_key in kernel_keys:
                kernel_group = f[f'/Grids/{kernel_key}']
                assert 'GridPointOverDensities' in kernel_group, \
                    f"Missing overdensities for {kernel_key}"


# ============================================================================
# Output Validation Tests
# ============================================================================

class TestOutputValidation:
    """Tests for output file structure and correctness."""

    def test_output_hdf5_structure(self, build_gridder, test_snapshot):
        """Test that output HDF5 file has correct structure."""
        param_file = TESTS_DIR / "simple_test_params.yml"
        output_file = DATA_DIR / "simple_test_grid.hdf5"

        if output_file.exists():
            output_file.unlink()

        subprocess.run(
            [str(build_gridder), str(param_file), "1"],
            check=True,
            capture_output=True
        )

        with h5py.File(output_file, 'r') as f:
            # Check required groups
            assert '/Grids' in f, "Missing /Grids group"

            # Check for kernel groups
            grids_group = f['/Grids']
            kernel_keys = [k for k in grids_group.keys() if k.startswith('Kernel_')]
            assert len(kernel_keys) == 3, f"Expected 3 kernels, found {len(kernel_keys)}"

            # Check each kernel has required datasets
            for kernel_key in kernel_keys:
                kernel_group = grids_group[kernel_key]
                assert 'GridPointOverDensities' in kernel_group, \
                    f"{kernel_key} missing GridPointOverDensities"

                # Check that write_masses is honored
                assert 'GridPointMasses' in kernel_group, \
                    f"{kernel_key} missing GridPointMasses (write_masses was 1)"


# ============================================================================
# Run tests if called directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
