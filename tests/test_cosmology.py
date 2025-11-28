#!/usr/bin/env python3
"""
Unit tests for cosmological mean density calculations.

Tests verify that the C++ cosmology implementation correctly computes
mean comoving density at various redshifts and with different cosmological
parameters.
"""

import subprocess
import pytest
import h5py
import numpy as np
from pathlib import Path
import yaml
import tempfile

# Try to import astropy for direct comparison tests
try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Get paths
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
GRIDDER_EXE = BUILD_DIR / "parent_gridder"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = TESTS_DIR / "data"


def create_test_snapshot(filepath, redshift, boxsize=10.0):
    """Create a minimal test snapshot at specified redshift"""
    with h5py.File(filepath, 'w') as f:
        # Header
        header = f.create_group('Header')
        header.attrs['Redshift'] = redshift
        header.attrs['BoxSize'] = np.array([boxsize, boxsize, boxsize])
        header.attrs['NumPart_Total'] = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint64)

        # Units (10^10 Msun, Mpc, km/s)
        units = f.create_group('Units')
        units.attrs['Unit mass in cgs (U_M)'] = 1.989e43
        units.attrs['Unit length in cgs (U_L)'] = 3.086e24
        units.attrs['Unit time in cgs (U_t)'] = 3.086e19

        # Cells metadata
        cells_meta = f.create_group('Cells/Meta-data')
        cells_meta.attrs['dimension'] = np.array([2, 2, 2], dtype=np.int32)
        cells_meta.attrs['size'] = np.array([boxsize/2, boxsize/2, boxsize/2])

        # One particle at center
        part1 = f.create_group('PartType1')
        part1.create_dataset('Coordinates', data=np.array([[boxsize/2, boxsize/2, boxsize/2]]))
        part1.create_dataset('Masses', data=np.array([1.0]))

        # Cell counts (one particle in central cell)
        f.create_dataset('Cells/Counts/PartType1', data=np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32))
        f.create_dataset('Cells/OffsetsInFile/PartType1', data=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32))


def create_test_params(filepath, snapshot_path, cosmology):
    """Create test parameter file with specified cosmology"""
    params = {
        'Kernels': {
            'nkernels': 1,
            'kernel_radius_1': 1.0
        },
        'Grid': {
            'type': 'uniform',
            'cdim': 2
        },
        'Cosmology': cosmology,
        'Tree': {
            'max_leaf_count': 200
        },
        'Input': {
            'filepath': str(snapshot_path),
            'placeholder': '0000'
        },
        'Output': {
            'filepath': str(filepath.parent) + '/',
            'basename': filepath.name,
            'write_masses': 0
        }
    }

    with open(filepath, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)


def run_gridder_and_get_density(snapshot_path, params_path, output_path):
    """Run gridder and extract mean density from output"""
    result = subprocess.run(
        [str(GRIDDER_EXE), str(params_path), "1"],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"Gridder failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Extract mean density from stdout
    for line in result.stdout.split('\n'):
        if 'Mean comoving density at z=' in line:
            # Parse: "Mean comoving density at z=X.XXXX: Y.YYYYYYe+XX 10^10 Msun/cMpc^3"
            parts = line.split(':')
            if len(parts) >= 2:
                density_str = parts[1].strip().split()[0]
                return float(density_str)

    raise RuntimeError(f"Could not extract mean density from output:\n{result.stdout}")


def calculate_expected_density(h, Omega_cdm, Omega_b, redshift):
    """
    Calculate expected mean COMOVING density using standard cosmological formula.

    ρ_comoving = ρ_crit(z=0) × Ω_m

    Note: In comoving coordinates, density does NOT evolve with redshift.
    The (1+z)³ factor converts to physical density, but SWIFT uses comoving coords.

    where ρ_crit(z=0) = 3H₀²/(8πG)

    Units: 10^10 Msun/cMpc^3
    """
    # Physical constants in internal units
    H0_kmsMpc = 100.0 * h  # km/s/Mpc
    G = 4.301744232015554e+01  # (10^10 Msun)^-1 Mpc (km/s)^2

    # Critical density today
    rho_crit_0 = (3.0 * H0_kmsMpc**2) / (8.0 * np.pi * G)

    # Total matter density parameter
    Omega_m = Omega_cdm + Omega_b

    # Mean COMOVING density (constant with redshift!)
    mean_density = rho_crit_0 * Omega_m

    return mean_density


class TestCosmologyCalculations:
    """Test suite for cosmological mean density calculations"""

    @pytest.fixture(scope="class")
    def test_workspace(self):
        """Create temporary workspace for tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            yield workspace

    def test_planck2018_z0(self, test_workspace):
        """Test Planck 2018 cosmology at z=0"""
        # Planck 2018 parameters
        cosmology = {
            'h': 0.6736,
            'Omega_cdm': 0.2607,
            'Omega_b': 0.0493
        }
        redshift = 0.0

        # Create test files
        snapshot_path = test_workspace / "snapshot_z0.hdf5"
        params_path = test_workspace / "params_z0.yml"
        output_path = test_workspace / "output_z0.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        # Run gridder and extract density
        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)

        # Calculate expected density
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        # Check agreement (within 0.01% - account for floating point precision)
        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch at z={redshift}: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_planck2018_z1(self, test_workspace):
        """Test Planck 2018 cosmology at z=1"""
        cosmology = {
            'h': 0.6736,
            'Omega_cdm': 0.2607,
            'Omega_b': 0.0493
        }
        redshift = 1.0

        snapshot_path = test_workspace / "snapshot_z1.hdf5"
        params_path = test_workspace / "params_z1.yml"
        output_path = test_workspace / "output_z1.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        # At z=1, density should be 8x higher than z=0 (since (1+z)³ = 8)
        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch at z={redshift}: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_high_redshift_z7(self, test_workspace):
        """Test at high redshift z=7 (relevant for JWST observations)"""
        cosmology = {
            'h': 0.6736,
            'Omega_cdm': 0.2607,
            'Omega_b': 0.0493
        }
        redshift = 7.0

        snapshot_path = test_workspace / "snapshot_z7.hdf5"
        params_path = test_workspace / "params_z7.yml"
        output_path = test_workspace / "output_z7.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        # At z=7, density should be (1+7)³ = 512x higher than z=0
        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch at z={redshift}: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_wmap9_cosmology(self, test_workspace):
        """Test WMAP9 cosmology (different from Planck)"""
        # WMAP9 parameters
        cosmology = {
            'h': 0.697,
            'Omega_cdm': 0.233,
            'Omega_b': 0.0463
        }
        redshift = 2.0

        snapshot_path = test_workspace / "snapshot_wmap9.hdf5"
        params_path = test_workspace / "params_wmap9.yml"
        output_path = test_workspace / "output_wmap9.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch with WMAP9 cosmology: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_low_omega_matter(self, test_workspace):
        """Test with artificially low matter density"""
        cosmology = {
            'h': 0.7,
            'Omega_cdm': 0.15,
            'Omega_b': 0.03
        }
        redshift = 0.5

        snapshot_path = test_workspace / "snapshot_low_om.hdf5"
        params_path = test_workspace / "params_low_om.yml"
        output_path = test_workspace / "output_low_om.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch with low Omega_m: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_high_h(self, test_workspace):
        """Test with high Hubble constant"""
        cosmology = {
            'h': 0.9,  # Unrealistically high but good test
            'Omega_cdm': 0.25,
            'Omega_b': 0.05
        }
        redshift = 1.5

        snapshot_path = test_workspace / "snapshot_high_h.hdf5"
        params_path = test_workspace / "params_high_h.yml"
        output_path = test_workspace / "output_high_h.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
        expected_density = calculate_expected_density(
            cosmology['h'], cosmology['Omega_cdm'], cosmology['Omega_b'], redshift
        )

        relative_error = abs(gridder_density - expected_density) / expected_density
        assert relative_error < 1e-4, \
            f"Mean density mismatch with high h: gridder={gridder_density:.6e}, expected={expected_density:.6e}"

    def test_density_constant_with_redshift(self, test_workspace):
        """Test that COMOVING density is constant with redshift"""
        cosmology = {
            'h': 0.6736,
            'Omega_cdm': 0.2607,
            'Omega_b': 0.0493
        }

        redshifts = [0.0, 1.0, 2.0, 5.0, 7.0]
        densities = []

        for z in redshifts:
            snapshot_path = test_workspace / f"snapshot_z{z}.hdf5"
            params_path = test_workspace / f"params_z{z}.yml"
            output_path = test_workspace / f"output_z{z}.hdf5"

            create_test_snapshot(snapshot_path, z)
            create_test_params(params_path, snapshot_path, cosmology)

            density = run_gridder_and_get_density(snapshot_path, params_path, output_path)
            densities.append(density)

        # Check that comoving density is constant with redshift (within 0.01%)
        reference_density = densities[0]
        for i, z in enumerate(redshifts):
            relative_diff = abs(densities[i] - reference_density) / reference_density
            assert relative_diff < 1e-4, \
                f"Comoving density should be constant: rho(z={z})={densities[i]:.6e}, rho(z=0)={reference_density:.6e}, diff={relative_diff:.6e}"

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_astropy_comparison(self, test_workspace):
        """Test that our calculation matches astropy exactly"""
        # Planck 2018 parameters
        cosmology = {
            'h': 0.6736,
            'Omega_cdm': 0.2607,
            'Omega_b': 0.0493
        }
        redshift = 2.0

        # Create test files
        snapshot_path = test_workspace / "snapshot_astropy.hdf5"
        params_path = test_workspace / "params_astropy.yml"
        output_path = test_workspace / "output_astropy.hdf5"

        create_test_snapshot(snapshot_path, redshift)
        create_test_params(params_path, snapshot_path, cosmology)

        # Get density from our C++ code
        gridder_density = run_gridder_and_get_density(snapshot_path, params_path, output_path)

        # Calculate with astropy
        h = cosmology['h']
        Omega_m = cosmology['Omega_cdm'] + cosmology['Omega_b']
        H0 = h * 100.0  # km/s/Mpc

        # Create astropy cosmology (flat ΛCDM with Omega_Lambda = 1 - Omega_m)
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        # Get critical density at z=0 in comoving coordinates
        rho_crit_0 = cosmo.critical_density(0)

        # Convert to our units: 10^10 Msun/Mpc^3
        # astropy gives g/cm^3, we need 10^10 Msun/Mpc^3
        Msun_in_g = 1.989e33
        Mpc_in_cm = 3.086e24
        rho_crit_0_our_units = rho_crit_0.to(u.g / u.cm**3).value * (Mpc_in_cm**3) / (1e10 * Msun_in_g)

        # Mean comoving density
        astropy_density = rho_crit_0_our_units * Omega_m

        # Check agreement (within 0.1% - account for different constants)
        relative_error = abs(gridder_density - astropy_density) / astropy_density
        assert relative_error < 1e-3, \
            f"Density mismatch vs astropy: gridder={gridder_density:.6e}, astropy={astropy_density:.6e}, error={relative_error:.6e}"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
