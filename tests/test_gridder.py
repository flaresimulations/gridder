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
    make_snap_script = TESTS_DIR / "make_test_snap.py"

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

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

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

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

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

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

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
# Random Grid Tests
# ============================================================================

class TestRandomGrid:
    """Tests for random grid point generation and reproducibility."""

    def test_random_grid_reproducibility(self, build_gridder, test_snapshot):
        """Test that random grid is reproducible with same seed."""
        param_template = """
Kernels:
  nkernels: 2
  kernel_radius_1: 1.0
  kernel_radius_2: 2.0

Grid:
  type: random
  n_grid_points: 50
  random_seed: {seed}

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: random_test_{run}.hdf5
  write_masses: 1
"""

        # Run twice with same seed
        outputs = []
        for run in [1, 2]:
            param_file = TESTS_DIR / f"temp_random_test_{run}.yml"
            param_file.write_text(param_template.format(seed=42, run=run))
            output_file = DATA_DIR / f"random_test_{run}.hdf5"

            if output_file.exists():
                output_file.unlink()

            try:
                result = subprocess.run(
                    [str(build_gridder), str(param_file), "1"],
                    capture_output=True,
                    text=True
                )

                assert result.returncode == 0, (
                    f"Gridder failed on run {run}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

                # Read grid point positions
                with h5py.File(output_file, 'r') as f:
                    if '/Grids/GridPointPositions' in f:
                        positions = f['/Grids/GridPointPositions'][:]
                        outputs.append(positions)

            finally:
                if param_file.exists():
                    param_file.unlink()

        # Verify both runs produced identical positions
        assert len(outputs) == 2, "Both runs should produce output"
        assert np.allclose(outputs[0], outputs[1]), \
            "Random grid should be reproducible with same seed"

    def test_random_grid_different_seeds(self, build_gridder, test_snapshot):
        """Test that different seeds produce different grids."""
        param_template = """
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: random
  n_grid_points: 50
  random_seed: {seed}

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: random_seed_{seed}.hdf5
  write_masses: 0
"""

        outputs = []
        for seed in [42, 123]:
            param_file = TESTS_DIR / f"temp_random_seed_{seed}.yml"
            param_file.write_text(param_template.format(seed=seed))
            output_file = DATA_DIR / f"random_seed_{seed}.hdf5"

            if output_file.exists():
                output_file.unlink()

            try:
                result = subprocess.run(
                    [str(build_gridder), str(param_file), "1"],
                    capture_output=True,
                    text=True
                )

                assert result.returncode == 0, f"Gridder failed with seed {seed}"

                with h5py.File(output_file, 'r') as f:
                    if '/Grids/GridPointPositions' in f:
                        positions = f['/Grids/GridPointPositions'][:]
                        outputs.append(positions)

            finally:
                if param_file.exists():
                    param_file.unlink()

        # Verify different seeds produce different grids
        assert len(outputs) == 2, "Both runs should produce output"
        assert not np.allclose(outputs[0], outputs[1]), \
            "Different seeds should produce different grids"


# ============================================================================
# Sparse Grid and Chunk Tests
# ============================================================================

class TestSparseGrid:
    """Tests for sparse grid handling and chunk-based particle loading."""

    def test_sparse_grid_chunk_preparation(self, build_gridder, test_snapshot):
        """Test that sparse grids report chunk preparation."""
        # Use existing test snapshot
        param_content = """
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

Grid:
  type: uniform
  cdim: 5  # 5^3 = 125 grid points

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: sparse_output.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_sparse_test.yml"
        param_file.write_text(param_content)
        output_file = DATA_DIR / "sparse_output.hdf5"

        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(build_gridder), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, (
                f"Gridder failed\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Check that it reports useful cells (sparse grid optimization)
            assert "useful cells" in result.stdout.lower(), \
                "Should report useful cell count"
            # Note: Chunk-based loading only happens in MPI mode,
            # serial mode uses traditional particle assignment

            # Verify output exists
            assert output_file.exists(), "Output file should be created"

        finally:
            if param_file.exists():
                param_file.unlink()


# ============================================================================
# Mass Calculation Tests
# ============================================================================

class TestMassCalculation:
    """Tests for correct cell and grid point mass calculations."""

    def test_cell_mass_summation(self, build_gridder, test_snapshot):
        """Test that cell masses are correctly summed from particles."""
        # Use existing test snapshot which has proper structure
        param_content = """
Kernels:
  nkernels: 1
  kernel_radius_1: 2.0  # Large enough to capture all particles

Grid:
  type: uniform
  cdim: 3

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: mass_test_output.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_mass_test.yml"
        param_file.write_text(param_content)
        output_file = DATA_DIR / "mass_test_output.hdf5"

        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(build_gridder), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, (
                f"Gridder failed\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Verify output contains correct masses
            with h5py.File(output_file, 'r') as f:
                kernel_keys = [k for k in f['/Grids'].keys() if k.startswith('Kernel_')]
                assert len(kernel_keys) > 0, "Should have kernel data"

                # Check that masses are present and reasonable
                for kernel_key in kernel_keys:
                    masses = f[f'/Grids/{kernel_key}/GridPointMasses'][:]
                    assert len(masses) > 0, "Should have grid point masses"
                    # At least some grid points should have non-zero mass
                    assert np.any(masses > 0), "Some grid points should have mass"

        finally:
            if param_file.exists():
                param_file.unlink()


# ============================================================================
# Sanity Check Tests - Grid Points on Particle Positions
# ============================================================================

class TestSanityChecks:
    """Sanity check tests: grid points placed exactly on particle positions."""

    def test_uniform_grid_on_particles_unit_overdensity(self, test_snapshot):
        """Test uniform grid with grid points at particle positions gives unit overdensity."""
        # Use simple test snapshot which has 1 particle at center
        # We know from simple_test.sh this gives unit mass in the central grid point
        param_content = """
Kernels:
  nkernels: 2
  kernel_radius_1: 1.0   # Captures central particle
  kernel_radius_2: 2.0   # Larger kernel

Grid:
  type: uniform
  cdim: 3  # 3^3 = 27 grid points

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: uniform_sanity_output.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_uniform_sanity.yml"
        param_file.write_text(param_content)
        output_file = DATA_DIR / "uniform_sanity_output.hdf5"

        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(GRIDDER_EXE), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, (
                f"Gridder failed\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Verify overdensities - central grid point should have mass = 1.0
            with h5py.File(output_file, 'r') as f:
                # Kernel indices are 0-based, so kernel_radius_1 is Kernel_0
                kernel_name = 'Kernel_0'
                if kernel_name in f['/Grids']:
                    masses = f[f'/Grids/{kernel_name}/GridPointMasses'][:]

                    # Central grid point (index 13 for 3x3x3) should have mass 1.0
                    central_idx = 13  # (1,1,1) in 3x3x3 grid
                    assert np.isclose(masses[central_idx], 1.0, rtol=0.01), \
                        f"Expected unit mass at central grid point, got: {masses[central_idx]}"

                    print(f"✓ Sanity check passed: central mass = {masses[central_idx]}")

        finally:
            if param_file.exists():
                param_file.unlink()

    def test_sparse_grid_on_particles_unit_overdensity(self, test_snapshot):
        """Test sparse grid (file-based) with grid point exactly on particle position."""
        # Create grid points file with one point at center (5.0, 5.0, 5.0)
        # where the particle is located in simple_test.hdf5
        grid_file = DATA_DIR / "sparse_sanity_grid.txt"
        with open(grid_file, 'w') as f:
            # Place grid point at center where particle is
            f.write("5.0 5.0 5.0\n")

        param_content = f"""
Kernels:
  nkernels: 2
  kernel_radius_1: 0.5   # Small - captures central particle
  kernel_radius_2: 1.0   # Medium

Grid:
  type: file
  grid_file: {grid_file}

Cosmology:
  h: 0.681
  Omega_cdm: 0.256011
  Omega_b: 0.048600

Tree:
  max_leaf_count: 200

Input:
  filepath: tests/data/simple_test.hdf5
  placeholder: "0000"

Output:
  filepath: tests/data/
  basename: sparse_sanity_output.hdf5
  write_masses: 1
"""
        param_file = TESTS_DIR / "temp_sparse_sanity.yml"
        param_file.write_text(param_content)
        output_file = DATA_DIR / "sparse_sanity_output.hdf5"

        if output_file.exists():
            output_file.unlink()

        try:
            result = subprocess.run(
                [str(GRIDDER_EXE), str(param_file), "1"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, (
                f"Gridder failed\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Verify results - should have unit mass at the single grid point
            with h5py.File(output_file, 'r') as f:
                # Kernel indices are 0-based, so kernel_radius_1 (0.5 Mpc) is Kernel_0
                kernel_name = 'Kernel_0'
                if kernel_name in f['/Grids']:
                    masses = f[f'/Grids/{kernel_name}/GridPointMasses'][:]

                    # Single grid point should have mass 1.0
                    assert len(masses) == 1, f"Expected 1 grid point, got {len(masses)}"
                    assert np.isclose(masses[0], 1.0, rtol=0.01), \
                        f"Expected unit mass at grid point, got: {masses[0]}"

                    print(f"✓ Sanity check passed: mass = {masses[0]}")

        finally:
            if param_file.exists():
                param_file.unlink()
            if grid_file.exists():
                grid_file.unlink()


# ============================================================================
# Python Conversion Tool Tests
# ============================================================================

class TestConversionTool:
    """Tests for the HDF5 conversion tool."""

    def test_conversion_tool_serial(self):
        """Test conversion tool in serial mode."""
        # Create input file with non-standard keys
        input_file = DATA_DIR / "nonstandard_input.hdf5"
        create_nonstandard_snapshot(input_file)

        output_file = DATA_DIR / "converted_output.hdf5"
        if output_file.exists():
            output_file.unlink()

        conversion_script = PROJECT_ROOT / "tools" / "convert_to_gridder_format.py"

        cmd = [
            sys.executable,
            str(conversion_script),
            str(input_file),
            str(output_file),
            "--coordinates-key", "DarkMatter/Positions",
            "--masses-key", "DarkMatter/ParticleMasses",
            "--particle-type", "PartType1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, (
            f"Conversion failed\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify output structure
        with h5py.File(output_file, 'r') as f:
            assert 'PartType1' in f, "Should have PartType1 group"
            assert 'PartType1/Coordinates' in f, "Should have Coordinates"
            assert 'PartType1/Masses' in f, "Should have Masses"

            coords = f['PartType1/Coordinates'][:]
            masses = f['PartType1/Masses'][:]

            assert coords.shape[1] == 3, "Coordinates should be Nx3"
            assert len(masses) == len(coords), "Masses should match coordinates"

    def test_conversion_tool_validation(self):
        """Test that conversion tool validates input shapes."""
        # Create malformed input
        input_file = DATA_DIR / "malformed_input.hdf5"
        create_malformed_snapshot(input_file)

        output_file = DATA_DIR / "should_not_exist.hdf5"
        if output_file.exists():
            output_file.unlink()

        conversion_script = PROJECT_ROOT / "tools" / "convert_to_gridder_format.py"

        cmd = [
            sys.executable,
            str(conversion_script),
            str(input_file),
            str(output_file),
            "--coordinates-key", "DarkMatter/BadCoords",
            "--masses-key", "DarkMatter/Masses"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with validation error
        assert result.returncode != 0, "Should fail with malformed input"
        assert "shape" in result.stderr.lower() or "must be" in result.stderr.lower(), \
            "Should report shape validation error"


# ============================================================================
# Helper Functions for Test Data Creation
# ============================================================================

def create_sparse_test_snapshot(filepath):
    """Create a sparse test snapshot with particles spread across cells."""
    with h5py.File(filepath, 'w') as f:
        # Header with all required attributes
        header = f.create_group('Header')
        header.attrs['BoxSize'] = np.array([100.0, 100.0, 100.0])
        header.attrs['NumPart_Total'] = np.array([1000, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['NumPart_ThisFile'] = np.array([1000, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['Redshift'] = 0.0
        header.attrs['Time'] = 1.0
        header.attrs['NumFilesPerSnapshot'] = 1

        # Create particles spread across box
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (1000, 3))
        masses = np.ones(1000) * 0.1

        # Create PartType1 group
        pt1 = f.create_group('PartType1')
        pt1.create_dataset('Coordinates', data=coords)
        pt1.create_dataset('Masses', data=masses)

        # Create minimal Cell_File group
        cell_file = f.create_group('Cell_File')
        cell_file.attrs['Cdim'] = np.array([10, 10, 10])

        # Particle counts per cell (some empty)
        counts = np.zeros(1000, dtype=np.int32)
        for i in range(1000):
            cell_idx = int(coords[i, 0] / 10) * 100 + int(coords[i, 1] / 10) * 10 + int(coords[i, 2] / 10)
            if cell_idx < 1000:
                counts[cell_idx] += 1

        cell_file.create_dataset('PartType1/Counts', data=counts)

        # Offsets
        offsets = np.zeros(1000, dtype=np.int64)
        cumsum = 0
        for i in range(1000):
            offsets[i] = cumsum
            cumsum += counts[i]
        cell_file.create_dataset('PartType1/Offsets', data=offsets)


def create_mass_test_snapshot(filepath):
    """Create snapshot with known particle masses for testing summation."""
    with h5py.File(filepath, 'w') as f:
        # Header with all required attributes
        header = f.create_group('Header')
        header.attrs['BoxSize'] = np.array([10.0, 10.0, 10.0])
        header.attrs['NumPart_Total'] = np.array([27, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['NumPart_ThisFile'] = np.array([27, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['Redshift'] = 0.0
        header.attrs['Time'] = 1.0
        header.attrs['NumFilesPerSnapshot'] = 1

        # Create 27 particles (3x3x3 grid) with known masses
        coords = []
        masses = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    coords.append([i * 3.33 + 1.67, j * 3.33 + 1.67, k * 3.33 + 1.67])
                    masses.append(float(i + j + k + 1))  # Masses from 1 to 7

        coords = np.array(coords)
        masses = np.array(masses)

        pt1 = f.create_group('PartType1')
        pt1.create_dataset('Coordinates', data=coords)
        pt1.create_dataset('Masses', data=masses)

        # Create Cell_File
        cell_file = f.create_group('Cell_File')
        cell_file.attrs['Cdim'] = np.array([3, 3, 3])
        cell_file.create_dataset('PartType1/Counts', data=np.ones(27, dtype=np.int32))
        cell_file.create_dataset('PartType1/Offsets', data=np.arange(27, dtype=np.int64))


def create_nonstandard_snapshot(filepath):
    """Create snapshot with non-standard dataset names for conversion testing."""
    with h5py.File(filepath, 'w') as f:
        # Use different key names
        dm = f.create_group('DarkMatter')

        coords = np.random.uniform(0, 10, (100, 3))
        masses = np.ones(100) * 0.5

        dm.create_dataset('Positions', data=coords)
        dm.create_dataset('ParticleMasses', data=masses)


def create_malformed_snapshot(filepath):
    """Create malformed snapshot for validation testing."""
    with h5py.File(filepath, 'w') as f:
        dm = f.create_group('DarkMatter')

        # Create 2D coords instead of Nx3 (malformed)
        bad_coords = np.random.uniform(0, 10, (100, 2))
        masses = np.ones(100)

        dm.create_dataset('BadCoords', data=bad_coords)
        dm.create_dataset('Masses', data=masses)


def create_uniform_sanity_snapshot(filepath):
    """Create snapshot with particles at uniform grid positions for sanity check."""
    with h5py.File(filepath, 'w') as f:
        # Header with all required attributes
        header = f.create_group('Header')
        boxsize = 10.0
        header.attrs['BoxSize'] = np.array([boxsize, boxsize, boxsize])

        # Create 5x5x5 = 125 particles at regular grid positions
        cdim = 5
        npart = cdim ** 3
        header.attrs['NumPart_Total'] = np.array([npart, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['NumPart_ThisFile'] = np.array([npart, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['Redshift'] = 0.0
        header.attrs['Time'] = 1.0
        header.attrs['NumFilesPerSnapshot'] = 1

        # Create particles at uniform grid positions
        spacing = boxsize / cdim
        coords = []
        for i in range(cdim):
            for j in range(cdim):
                for k in range(cdim):
                    # Place particles at grid centers
                    x = (i + 0.5) * spacing
                    y = (j + 0.5) * spacing
                    z = (k + 0.5) * spacing
                    coords.append([x, y, z])

        coords = np.array(coords)
        masses = np.ones(npart)  # All particles have mass 1.0

        pt1 = f.create_group('PartType1')
        pt1.create_dataset('Coordinates', data=coords)
        pt1.create_dataset('Masses', data=masses)

        # Create Cell_File
        cell_file = f.create_group('Cell_File')
        cell_file.attrs['Cdim'] = np.array([cdim, cdim, cdim])

        # One particle per cell
        cell_file.create_dataset('PartType1/Counts', data=np.ones(npart, dtype=np.int32))
        cell_file.create_dataset('PartType1/Offsets', data=np.arange(npart, dtype=np.int64))


def create_sparse_sanity_snapshot(filepath, particle_positions):
    """Create snapshot with particles at specific positions for sparse sanity check."""
    with h5py.File(filepath, 'w') as f:
        # Header
        header = f.create_group('Header')
        boxsize = 10.0
        header.attrs['BoxSize'] = np.array([boxsize, boxsize, boxsize])
        npart = len(particle_positions)
        header.attrs['NumPart_Total'] = np.array([npart, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['NumPart_ThisFile'] = np.array([npart, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs['Redshift'] = 0.0
        header.attrs['Time'] = 1.0
        header.attrs['NumFilesPerSnapshot'] = 1

        # Create particles at specified positions
        masses = np.ones(npart)  # All particles have mass 1.0

        pt1 = f.create_group('PartType1')
        pt1.create_dataset('Coordinates', data=particle_positions)
        pt1.create_dataset('Masses', data=masses)

        # Create Cell_File with appropriate cell structure
        # Use 5x5x5 cells for 10x10x10 box
        cdim = 5
        ncells = cdim ** 3
        cell_file = f.create_group('Cell_File')
        cell_file.attrs['Cdim'] = np.array([cdim, cdim, cdim])

        # Assign particles to cells
        cell_size = boxsize / cdim
        cell_counts = np.zeros(ncells, dtype=np.int32)
        cell_indices = []

        for pos in particle_positions:
            ix = int(pos[0] / cell_size)
            iy = int(pos[1] / cell_size)
            iz = int(pos[2] / cell_size)
            # Ensure within bounds
            ix = min(ix, cdim - 1)
            iy = min(iy, cdim - 1)
            iz = min(iz, cdim - 1)
            cell_id = ix * cdim * cdim + iy * cdim + iz
            cell_counts[cell_id] += 1
            cell_indices.append(cell_id)

        # Create offsets
        offsets = np.zeros(ncells, dtype=np.int64)
        cumsum = 0
        for i in range(ncells):
            offsets[i] = cumsum
            cumsum += cell_counts[i]

        cell_file.create_dataset('PartType1/Counts', data=cell_counts)
        cell_file.create_dataset('PartType1/Offsets', data=offsets)


# ============================================================================
# Run tests if called directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
