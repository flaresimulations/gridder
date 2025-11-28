#!/usr/bin/env python3
"""
Test suite for convert_to_gridder_format.py

Tests the conversion script's ability to:
1. Create proper cell structures
2. Handle various input formats
3. Produce files that the gridder can read successfully
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Get paths
PROJECT_ROOT = Path(__file__).parent.parent
CONVERSION_SCRIPT = PROJECT_ROOT / "tools" / "convert_to_gridder_format.py"
GRIDDER_EXE = PROJECT_ROOT / "build" / "parent_gridder"


@pytest.fixture(scope="session")
def build_gridder():
    """Ensure gridder is built before running tests."""
    if not GRIDDER_EXE.exists():
        print("Building gridder...")
        subprocess.run(
            ["cmake", "-B", str(PROJECT_ROOT / "build")], cwd=PROJECT_ROOT, check=True
        )
        subprocess.run(
            ["cmake", "--build", str(PROJECT_ROOT / "build")],
            cwd=PROJECT_ROOT,
            check=True,
        )
    assert GRIDDER_EXE.exists(), f"Gridder executable not found at {GRIDDER_EXE}"
    return GRIDDER_EXE


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_arbitrary_snapshot(
    filepath,
    coords_key="Coords",
    masses_key="Masses",
    npart=1000,
    boxsize=10.0,
    include_header=True,
):
    """
    Create a test snapshot with arbitrary key names.

    Args:
        filepath: Output file path
        coords_key: Key name for coordinates
        masses_key: Key name for masses
        npart: Number of particles
        boxsize: Box size
        include_header: Whether to include Header group
    """
    np.random.seed(42)
    coords = np.random.uniform(0, boxsize, (npart, 3))
    masses = np.ones(npart) * 0.5

    with h5py.File(filepath, "w") as f:
        # Write particles with arbitrary keys
        f.create_dataset(coords_key, data=coords)
        f.create_dataset(masses_key, data=masses)

        # Add header if requested
        if include_header:
            header = f.create_group("Header")
            header.attrs["BoxSize"] = np.array([boxsize, boxsize, boxsize])
            header.attrs["NumPart_Total"] = np.array(
                [0, npart, 0, 0, 0, 0], dtype=np.uint64
            )
            header.attrs["Redshift"] = 0.0


def create_uniform_snapshot(filepath, npart_per_dim=5, boxsize=10.0):
    """
    Create a snapshot with particles on a uniform grid.

    Args:
        filepath: Output file path
        npart_per_dim: Number of particles per dimension
        boxsize: Box size
    """
    spacing = boxsize / npart_per_dim
    coords = []
    for i in range(npart_per_dim):
        for j in range(npart_per_dim):
            for k in range(npart_per_dim):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                z = (k + 0.5) * spacing
                coords.append([x, y, z])

    coords = np.array(coords)
    npart = len(coords)
    masses = np.ones(npart)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("DarkMatter/Positions", data=coords)
        f.create_dataset("DarkMatter/Masses", data=masses)

        header = f.create_group("Header")
        header.attrs["BoxSize"] = np.array([boxsize, boxsize, boxsize])
        header.attrs["NumPart_Total"] = np.array(
            [0, npart, 0, 0, 0, 0], dtype=np.uint64
        )
        header.attrs["Redshift"] = 0.0


def create_noncubic_snapshot(filepath, boxsize_xyz, npart=500):
    """Create a snapshot with non-cubic box."""
    np.random.seed(123)
    coords = np.random.uniform([0, 0, 0], boxsize_xyz, (npart, 3))
    masses = np.ones(npart) * 2.0

    with h5py.File(filepath, "w") as f:
        f.create_dataset("PartType1/Coordinates", data=coords)
        f.create_dataset("PartType1/Masses", data=masses)

        header = f.create_group("Header")
        header.attrs["BoxSize"] = np.array(boxsize_xyz)
        header.attrs["NumPart_Total"] = np.array(
            [0, npart, 0, 0, 0, 0], dtype=np.uint64
        )
        header.attrs["Redshift"] = 0.0


def verify_cell_structure(hdf_file, expected_npart, cdim):
    """
    Verify that an HDF5 file has correct cell structure.

    Args:
        hdf_file: Path to HDF5 file
        expected_npart: Expected total number of particles
        cdim: Expected cell dimension

    Returns:
        True if structure is valid, raises AssertionError otherwise
    """
    with h5py.File(hdf_file, "r") as f:
        # Check required groups exist
        assert "/PartType1" in f, "Missing PartType1 group"
        assert "/Cells" in f, "Missing Cells group"

        # Check particle data
        assert "/PartType1/Coordinates" in f, "Missing Coordinates dataset"
        assert "/PartType1/Masses" in f, "Missing Masses dataset"

        coords = f["/PartType1/Coordinates"][:]
        masses = f["/PartType1/Masses"][:]

        assert coords.shape == (expected_npart, 3), (
            f"Wrong coordinates shape: {coords.shape} vs expected ({expected_npart}, 3)"
        )
        assert masses.shape == (expected_npart,), (
            f"Wrong masses shape: {masses.shape} vs expected ({expected_npart},)"
        )

        # Check cell metadata
        assert "/Cells/Meta-data" in f, "Missing Cells/Meta-data group"
        metadata = f["/Cells/Meta-data"]

        assert "dimension" in metadata.attrs, "Missing dimension attribute"
        assert "size" in metadata.attrs, "Missing size attribute"

        dimension = metadata.attrs["dimension"]
        assert np.array_equal(dimension, [cdim, cdim, cdim]), (
            f"Wrong dimension: {dimension} vs expected [{cdim}, {cdim}, {cdim}]"
        )

        # Check cell counts and offsets
        assert "/Cells/Counts/PartType1" in f, "Missing Counts dataset"
        assert "/Cells/OffsetsInFile/PartType1" in f, "Missing OffsetsInFile dataset"

        cell_counts = f["/Cells/Counts/PartType1"][:]
        cell_offsets = f["/Cells/OffsetsInFile/PartType1"][:]

        ncells = cdim**3
        assert cell_counts.shape == (ncells,), (
            f"Wrong cell_counts shape: {cell_counts.shape} vs expected ({ncells},)"
        )
        assert cell_offsets.shape == (ncells,), (
            f"Wrong cell_offsets shape: {cell_offsets.shape} vs expected ({ncells},)"
        )

        # Verify counts sum to total particles
        assert np.sum(cell_counts) == expected_npart, (
            f"Cell counts sum to {np.sum(cell_counts)}, expected {expected_npart}"
        )

        # Verify offsets are cumulative
        expected_offsets = np.zeros(ncells, dtype=np.int64)
        expected_offsets[1:] = np.cumsum(cell_counts[:-1])
        assert np.array_equal(cell_offsets, expected_offsets), (
            "Cell offsets are not cumulative"
        )

        # Verify particles are sorted by cell
        # Recompute cell assignments
        cell_size = metadata.attrs["size"]
        i = np.floor(coords[:, 0] / cell_size[0]).astype(np.int32)
        j = np.floor(coords[:, 1] / cell_size[1]).astype(np.int32)
        k = np.floor(coords[:, 2] / cell_size[2]).astype(np.int32)
        i = np.clip(i, 0, cdim - 1)
        j = np.clip(j, 0, cdim - 1)
        k = np.clip(k, 0, cdim - 1)
        cell_indices = k + j * cdim + i * cdim * cdim

        # Check that particles are sorted
        for idx in range(1, len(cell_indices)):
            assert cell_indices[idx] >= cell_indices[idx - 1], (
                f"Particles not sorted by cell at index {idx}"
            )

    return True


class TestConversion:
    """Test conversion script functionality."""

    def test_uniform_distribution(self, build_gridder, temp_dir):
        """Test conversion with uniform particle distribution."""
        input_file = temp_dir / "uniform_input.hdf5"
        output_file = temp_dir / "uniform_output.hdf5"

        # Create test input
        npart_per_dim = 5
        npart = npart_per_dim**3
        create_uniform_snapshot(input_file, npart_per_dim=npart_per_dim)

        # Run conversion
        cmd = [
            sys.executable,
            str(CONVERSION_SCRIPT),
            str(input_file),
            str(output_file),
            "--coordinates-key",
            "DarkMatter/Positions",
            "--masses-key",
            "DarkMatter/Masses",
            "--copy-header",
            "--cdim",
            "4",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify output structure
        verify_cell_structure(output_file, npart, cdim=4)

        # Test with gridder
        param_file = temp_dir / "test_params.yml"
        param_content = f"""
Kernels:
  nkernels: 1
  kernel_radius_1: 1.0

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
  filepath: {output_file}

Output:
  filepath: {temp_dir}/
  basename: gridded_output.hdf5
  write_masses: 1
"""
        param_file.write_text(param_content)

        result = subprocess.run(
            [str(build_gridder), str(param_file), "1"], capture_output=True, text=True
        )

        assert result.returncode == 0, (
            f"Gridder failed on converted file:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_random_distribution(self, build_gridder, temp_dir):
        """Test conversion with random particle distribution."""
        input_file = temp_dir / "random_input.hdf5"
        output_file = temp_dir / "random_output.hdf5"

        # Create test input with arbitrary keys
        npart = 1000
        create_arbitrary_snapshot(
            input_file,
            coords_key="MyCoordinates",
            masses_key="MyMasses",
            npart=npart,
            boxsize=20.0,
        )

        # Run conversion
        cmd = [
            sys.executable,
            str(CONVERSION_SCRIPT),
            str(input_file),
            str(output_file),
            "--coordinates-key",
            "MyCoordinates",
            "--masses-key",
            "MyMasses",
            "--copy-header",
            "--cdim",
            "10",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify output structure
        verify_cell_structure(output_file, npart, cdim=10)

    def test_noncubic_box(self, build_gridder, temp_dir):
        """Test conversion with non-cubic box."""
        input_file = temp_dir / "noncubic_input.hdf5"
        output_file = temp_dir / "noncubic_output.hdf5"

        # Create test input with non-cubic box
        npart = 500
        boxsize_xyz = [10.0, 20.0, 15.0]
        create_noncubic_snapshot(input_file, boxsize_xyz, npart)

        # Run conversion
        cmd = [
            sys.executable,
            str(CONVERSION_SCRIPT),
            str(input_file),
            str(output_file),
            "--coordinates-key",
            "PartType1/Coordinates",
            "--masses-key",
            "PartType1/Masses",
            "--copy-header",
            "--cdim",
            "8",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify output structure
        verify_cell_structure(output_file, npart, cdim=8)

        # Verify cell sizes match non-cubic box
        with h5py.File(output_file, "r") as f:
            cell_size = f["/Cells/Meta-data"].attrs["size"]
            expected_size = np.array(boxsize_xyz) / 8
            assert np.allclose(cell_size, expected_size), (
                f"Cell sizes incorrect: {cell_size} vs expected {expected_size}"
            )

    def test_boundary_particles(self, build_gridder, temp_dir):
        """Test handling of particles at box boundaries."""
        input_file = temp_dir / "boundary_input.hdf5"
        output_file = temp_dir / "boundary_output.hdf5"

        boxsize = 10.0

        # Create particles at boundaries and random interior
        np.random.seed(789)
        boundary_coords = np.array(
            [
                [0.0, 5.0, 5.0],  # x=0
                [10.0, 5.0, 5.0],  # x=boxsize
                [5.0, 0.0, 5.0],  # y=0
                [5.0, 10.0, 5.0],  # y=boxsize
                [5.0, 5.0, 0.0],  # z=0
                [5.0, 5.0, 10.0],  # z=boxsize
                [0.0, 0.0, 0.0],  # corner
                [10.0, 10.0, 10.0],  # opposite corner
            ]
        )
        interior_coords = np.random.uniform(0.1, 9.9, (100, 3))
        coords = np.vstack([boundary_coords, interior_coords])
        npart = len(coords)
        masses = np.ones(npart)

        with h5py.File(input_file, "w") as f:
            f.create_dataset("Coordinates", data=coords)
            f.create_dataset("Masses", data=masses)

            header = f.create_group("Header")
            header.attrs["BoxSize"] = np.array([boxsize, boxsize, boxsize])
            header.attrs["NumPart_Total"] = np.array(
                [0, npart, 0, 0, 0, 0], dtype=np.uint64
            )
            header.attrs["Redshift"] = 0.0

        # Run conversion
        cmd = [
            sys.executable,
            str(CONVERSION_SCRIPT),
            str(input_file),
            str(output_file),
            "--coordinates-key",
            "Coordinates",
            "--masses-key",
            "Masses",
            "--copy-header",
            "--cdim",
            "5",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify output structure
        verify_cell_structure(output_file, npart, cdim=5)

    def test_manual_boxsize(self, build_gridder, temp_dir):
        """Test specifying BoxSize manually."""
        input_file = temp_dir / "no_header_input.hdf5"
        output_file = temp_dir / "no_header_output.hdf5"

        # Create input without header
        npart = 200
        boxsize = 50.0
        np.random.seed(456)
        coords = np.random.uniform(0, boxsize, (npart, 3))
        masses = np.ones(npart) * 1.5

        with h5py.File(input_file, "w") as f:
            f.create_dataset("Positions", data=coords)
            f.create_dataset("ParticleMasses", data=masses)
            # No Header group!

        # Run conversion with manual BoxSize
        cmd = [
            sys.executable,
            str(CONVERSION_SCRIPT),
            str(input_file),
            str(output_file),
            "--coordinates-key",
            "Positions",
            "--masses-key",
            "ParticleMasses",
            "--boxsize",
            "50.0",
            "50.0",
            "50.0",
            "--cdim",
            "8",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify output structure
        verify_cell_structure(output_file, npart, cdim=8)

    def test_different_cdim_values(self, build_gridder, temp_dir):
        """Test various cell dimension values."""
        for cdim in [4, 10, 16, 32]:
            input_file = temp_dir / f"cdim{cdim}_input.hdf5"
            output_file = temp_dir / f"cdim{cdim}_output.hdf5"

            npart = 500
            create_arbitrary_snapshot(input_file, npart=npart, boxsize=10.0)

            cmd = [
                sys.executable,
                str(CONVERSION_SCRIPT),
                str(input_file),
                str(output_file),
                "--coordinates-key",
                "Coords",
                "--masses-key",
                "Masses",
                "--copy-header",
                "--cdim",
                str(cdim),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, (
                f"Conversion failed for cdim={cdim}: {result.stderr}"
            )

            verify_cell_structure(output_file, npart, cdim=cdim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
