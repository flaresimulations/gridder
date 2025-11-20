#!/usr/bin/env python3
"""
Comprehensive test suite for file-based gridding functionality.

This tests the critical case where grid points are loaded from a file,
which is the workflow that's failing for halo center gridding.
"""

import argparse
import h5py
import numpy as np
import os
import subprocess
import sys
import tempfile


class FileGriddingTest:
    """Test suite for file-based gridding"""

    def __init__(self, executable, test_dir="tests/data"):
        self.executable = executable
        self.test_dir = test_dir
        os.makedirs(test_dir, exist_ok=True)

        # Test configuration
        self.box_size = 10.0  # Mpc/h
        self.n_particles = 1000
        self.kernel_radii = [0.5, 2.0, 5.0]

    def create_test_snapshot(self, filename):
        """Create a simple test snapshot with particles in clusters"""
        print(f"Creating test snapshot: {filename}")

        with h5py.File(filename, 'w') as f:
            # Header
            header = f.create_group('Header')
            header.attrs['BoxSize'] = self.box_size
            header.attrs['NumPart_Total'] = [0, self.n_particles, 0, 0, 0, 0]
            header.attrs['NumPart_ThisFile'] = [0, self.n_particles, 0, 0, 0, 0]
            header.attrs['Redshift'] = 7.0

            # Units
            units = f.create_group('Units')
            units.attrs['Unit mass in cgs (U_M)'] = 1.989e43  # 10^10 Msun
            units.attrs['Unit length in cgs (U_L)'] = 3.086e24  # Mpc

            # Cosmology
            cosmo = f.create_group('Cosmology')
            cosmo.attrs['Critical density [internal units]'] = 1.0

            # Cells metadata (required by gridder)
            cells = f.create_group('Cells')
            cells_meta = cells.create_group('Meta-data')
            cells_meta.attrs['dimension'] = np.array([2, 2, 2], dtype=np.int32)
            cells_meta.attrs['nr_cells'] = 8
            cells_meta.attrs['size'] = self.box_size

            # Cells Centres
            cell_width = self.box_size / 2.0
            centres = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        centres.append([
                            (i + 0.5) * cell_width,
                            (j + 0.5) * cell_width,
                            (k + 0.5) * cell_width
                        ])
            cells.create_dataset('Centres', data=np.array(centres))

            # Create Counts and OffsetsInFile groups (not datasets!)
            counts_grp = cells.create_group('Counts')
            offsets_grp = cells.create_group('OffsetsInFile')

            # Will be filled after we know particle positions
            counts_grp.create_dataset('PartType1', data=np.zeros(8, dtype=np.int32))
            offsets_grp.create_dataset('PartType1', data=np.zeros(8, dtype=np.int32))

            # Create particle distributions: 5 clusters
            n_clusters = 5
            particles_per_cluster = self.n_particles // n_clusters
            cluster_centers = [
                [2.0, 2.0, 2.0],
                [5.0, 5.0, 5.0],
                [8.0, 2.0, 5.0],
                [2.0, 8.0, 5.0],
                [5.0, 5.0, 8.0],
            ]

            positions = []
            masses = []

            for center in cluster_centers:
                # Gaussian distribution around center
                cluster_pos = np.random.normal(
                    loc=center,
                    scale=0.3,  # Tight cluster
                    size=(particles_per_cluster, 3)
                )
                # Wrap to box
                cluster_pos = np.mod(cluster_pos, self.box_size)
                positions.append(cluster_pos)
                masses.extend([1.0] * particles_per_cluster)

            # Add remaining particles uniformly
            remainder = self.n_particles - (n_clusters * particles_per_cluster)
            if remainder > 0:
                uniform_pos = np.random.uniform(
                    0, self.box_size, size=(remainder, 3)
                )
                positions.append(uniform_pos)
                masses.extend([1.0] * remainder)

            positions = np.vstack(positions)
            masses = np.array(masses)

            # Compute cell counts for PartType1
            cell_counts = np.zeros(8, dtype=np.int32)
            cell_offsets = np.zeros(8, dtype=np.int32)

            # Assign particles to cells
            for i, pos in enumerate(positions):
                ix = int(pos[0] / cell_width)
                iy = int(pos[1] / cell_width)
                iz = int(pos[2] / cell_width)
                # Ensure within bounds
                ix = max(0, min(1, ix))
                iy = max(0, min(1, iy))
                iz = max(0, min(1, iz))
                cell_id = ix * 4 + iy * 2 + iz
                cell_counts[cell_id] += 1

            # Set offsets (cumulative)
            offset = 0
            for cell_id in range(8):
                cell_offsets[cell_id] = offset
                offset += cell_counts[cell_id]

            # Update cell datasets
            f['Cells/Counts/PartType1'][:] = cell_counts
            f['Cells/OffsetsInFile/PartType1'][:] = cell_offsets

            # Write particle data
            part1 = f.create_group('PartType1')
            part1.create_dataset('Coordinates', data=positions)
            part1.create_dataset('Masses', data=masses)
            part1.create_dataset('ParticleIDs', data=np.arange(len(masses)))

        print(f"  Created {len(masses)} particles in {n_clusters} clusters")
        return cluster_centers

    def create_grid_file(self, filename, grid_points):
        """Create a text file with grid point coordinates"""
        print(f"Creating grid file: {filename}")

        with open(filename, 'w') as f:
            f.write("# Test grid point coordinates\n")
            f.write("# Format: x y z (Mpc/h)\n")
            f.write(f"# {len(grid_points)} grid points\n")
            f.write("\n")

            for point in grid_points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        print(f"  Wrote {len(grid_points)} grid points")

    def create_param_file(self, filename, snapshot_file, grid_file,
                         output_file):
        """Create a parameter file for the gridder"""
        print(f"Creating parameter file: {filename}")

        content = f"""# Test parameter file for file-based gridding
Kernels:
  nkernels: {len(self.kernel_radii)}
"""
        for i, radius in enumerate(self.kernel_radii, 1):
            content += f"  kernel_radius_{i}: {radius}\n"

        content += f"""
Grid:
  type: file
  grid_file: {grid_file}

Tree:
  max_leaf_count: 200

Input:
  filepath: {snapshot_file}

Output:
  filepath: {os.path.dirname(output_file)}/
  basename: {os.path.basename(output_file)}
  write_masses: 1
"""

        with open(filename, 'w') as f:
            f.write(content)

    def run_gridder(self, param_file, nthreads=4, verbosity=2):
        """Run the gridder executable"""
        print(f"\nRunning gridder: {self.executable}")
        print(f"  Params: {param_file}")
        print(f"  Threads: {nthreads}")
        print(f"  Verbosity: {verbosity}")

        cmd = [self.executable, param_file, str(nthreads), str(verbosity)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("  Gridder completed successfully")

            # Always print debug output (contains validation results)
            if result.stdout:
                print("\n--- Gridder Output ---")
                print(result.stdout)
            if result.stderr:
                print("\n--- Stderr ---")
                print(result.stderr)
            print("--- End Output ---\n")

            # Check for debug validation failures in output
            if '[DEBUG]' in result.stdout or '[DEBUG]' in result.stderr:
                debug_output = result.stdout + result.stderr
                if 'ERROR' in debug_output or 'FAILED' in debug_output:
                    print("\n!!! DEBUG CHECKS DETECTED ERRORS !!!")
                    print("See debug output above for details.\n")

            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Gridder failed with exit code {e.returncode}")
            print("\n--- Stdout ---")
            print(e.stdout)
            print("\n--- Stderr ---")
            print(e.stderr)
            return False

    def validate_output(self, output_file, expected_grid_points):
        """Validate the gridder output"""
        print(f"\nValidating output: {output_file}")

        if not os.path.exists(output_file):
            print(f"  ERROR: Output file not found: {output_file}")
            return False

        errors = []

        with h5py.File(output_file, 'r') as f:
            # Check basic structure
            required_groups = ['Header', 'Grids', 'Cells']
            for group in required_groups:
                if group not in f:
                    errors.append(f"Missing required group: {group}")

            if 'Grids' not in f:
                return False

            # Check kernels (filter out non-kernel datasets like GridPointPositions)
            all_keys = list(f['Grids'].keys())
            kernels = [k for k in all_keys if k.startswith('Kernel_')]
            print(f"  Found {len(kernels)} kernels: {kernels}")
            if len(all_keys) != len(kernels):
                print(f"  (Ignoring non-kernel datasets: {[k for k in all_keys if not k.startswith('Kernel_')]})")

            if len(kernels) != len(self.kernel_radii):
                errors.append(
                    f"Expected {len(self.kernel_radii)} kernels, "
                    f"found {len(kernels)}"
                )

            # Check each kernel
            for i, expected_radius in enumerate(self.kernel_radii):
                kernel_name = f'Kernel_{i}'

                if kernel_name not in f['Grids']:
                    errors.append(f"Missing kernel: {kernel_name}")
                    continue

                kernel_grp = f['Grids'][kernel_name]

                # Check kernel radius attribute
                if 'KernelRadius' in kernel_grp.attrs:
                    actual_radius = kernel_grp.attrs['KernelRadius']
                    if not np.isclose(actual_radius, expected_radius):
                        errors.append(
                            f"{kernel_name}: Expected radius {expected_radius}, "
                            f"got {actual_radius}"
                        )
                else:
                    errors.append(
                        f"{kernel_name}: Missing KernelRadius attribute"
                    )

                # Check datasets
                required_datasets = ['GridPointOverDensities', 'GridPointMasses']
                for dataset in required_datasets:
                    if dataset not in kernel_grp:
                        errors.append(
                            f"{kernel_name}: Missing dataset {dataset}"
                        )
                        continue

                    data = kernel_grp[dataset][:]
                    print(f"  {kernel_name}/{dataset}: shape={data.shape}, "
                          f"min={data.min():.3f}, max={data.max():.3f}, "
                          f"mean={data.mean():.3f}")

                    # Check for data quality issues
                    n_empty = np.sum(data == -1)
                    n_zero = np.sum(data == 0)
                    n_positive = np.sum(data > 0)

                    print(f"    Values: {n_positive} positive, "
                          f"{n_zero} zero, {n_empty} empty (-1)")

                    # CRITICAL CHECK: For halo center gridding, we expect
                    # most grid points to find particles
                    if dataset == 'GridPointMasses':
                        # Check for empty (-1) values
                        if n_empty > len(data) * 0.5:
                            errors.append(
                                f"{kernel_name}: More than 50% of grid points "
                                f"have no particles ({n_empty}/{len(data)}). "
                                f"This suggests a major problem!"
                            )
                        # CRITICAL: Check for zero values when we expect particles
                        # (cluster centers should have particles!)
                        if 'cluster' in output_file.lower() and n_zero > len(data) * 0.5:
                            errors.append(
                                f"CRITICAL BUG: {kernel_name}: Grid points at cluster "
                                f"centers found ZERO particles ({n_zero}/{len(data)}). "
                                f"Particles exist at these locations but gridder "
                                f"is not finding them!"
                            )

        if errors:
            print("\n  VALIDATION ERRORS:")
            for error in errors:
                print(f"    - {error}")
            return False

        print("  ✓ Validation passed")
        return True

    def test_cluster_centers(self):
        """
        Test 1: Grid points at cluster centers
        This is the critical test - grid points at known high-density regions
        """
        print("\n" + "="*60)
        print("TEST 1: Grid points at cluster centers")
        print("="*60)

        # Create test files
        snapshot_file = os.path.join(self.test_dir, "test_clusters.hdf5")
        grid_file = os.path.join(self.test_dir, "cluster_centers.txt")
        param_file = os.path.join(self.test_dir, "test_clusters.yml")
        output_file = os.path.join(self.test_dir, "cluster_output.hdf5")

        # Create snapshot with known cluster centers
        cluster_centers = self.create_test_snapshot(snapshot_file)

        # Create grid file with grid points AT cluster centers
        # These points MUST find particles
        self.create_grid_file(grid_file, cluster_centers)

        # Create parameter file
        self.create_param_file(param_file, snapshot_file, grid_file,
                              output_file)

        # Run gridder
        if not self.run_gridder(param_file):
            print("\n✗ TEST 1 FAILED: Gridder execution failed")
            return False

        # Validate output
        if not self.validate_output(output_file, cluster_centers):
            print("\n✗ TEST 1 FAILED: Output validation failed")
            return False

        print("\n✓ TEST 1 PASSED: Cluster center gridding works")
        return True

    def test_random_points(self):
        """
        Test 2: Random grid points throughout box
        """
        print("\n" + "="*60)
        print("TEST 2: Random grid points")
        print("="*60)

        snapshot_file = os.path.join(self.test_dir, "test_random.hdf5")
        grid_file = os.path.join(self.test_dir, "random_points.txt")
        param_file = os.path.join(self.test_dir, "test_random.yml")
        output_file = os.path.join(self.test_dir, "random_output.hdf5")

        # Create snapshot
        self.create_test_snapshot(snapshot_file)

        # Create random grid points
        n_grid = 50
        grid_points = np.random.uniform(0, self.box_size, size=(n_grid, 3))
        self.create_grid_file(grid_file, grid_points)

        # Create parameter file
        self.create_param_file(param_file, snapshot_file, grid_file,
                              output_file)

        # Run and validate
        if not self.run_gridder(param_file):
            print("\n✗ TEST 2 FAILED: Gridder execution failed")
            return False

        if not self.validate_output(output_file, grid_points):
            print("\n✗ TEST 2 FAILED: Output validation failed")
            return False

        print("\n✓ TEST 2 PASSED: Random point gridding works")
        return True

    def test_boundary_points(self):
        """
        Test 3: Grid points near box boundaries (periodic BC test)
        """
        print("\n" + "="*60)
        print("TEST 3: Boundary and periodic BC test")
        print("="*60)

        snapshot_file = os.path.join(self.test_dir, "test_boundary.hdf5")
        grid_file = os.path.join(self.test_dir, "boundary_points.txt")
        param_file = os.path.join(self.test_dir, "test_boundary.yml")
        output_file = os.path.join(self.test_dir, "boundary_output.hdf5")

        # Create snapshot
        self.create_test_snapshot(snapshot_file)

        # Create grid points at box edges and corners
        eps = 0.1  # Small offset from boundary
        grid_points = [
            # Corners
            [eps, eps, eps],
            [self.box_size - eps, eps, eps],
            [eps, self.box_size - eps, eps],
            [eps, eps, self.box_size - eps],
            [self.box_size - eps, self.box_size - eps, eps],
            [self.box_size - eps, eps, self.box_size - eps],
            [eps, self.box_size - eps, self.box_size - eps],
            [self.box_size - eps, self.box_size - eps, self.box_size - eps],
            # Face centers
            [self.box_size / 2, eps, self.box_size / 2],
            [self.box_size / 2, self.box_size - eps, self.box_size / 2],
            [eps, self.box_size / 2, self.box_size / 2],
            [self.box_size - eps, self.box_size / 2, self.box_size / 2],
        ]
        self.create_grid_file(grid_file, grid_points)

        # Create parameter file
        self.create_param_file(param_file, snapshot_file, grid_file,
                              output_file)

        # Run and validate
        if not self.run_gridder(param_file):
            print("\n✗ TEST 3 FAILED: Gridder execution failed")
            return False

        if not self.validate_output(output_file, grid_points):
            print("\n✗ TEST 3 FAILED: Output validation failed")
            return False

        print("\n✓ TEST 3 PASSED: Boundary point gridding works")
        return True

    def run_all_tests(self):
        """Run all tests and report results"""
        print("\n" + "="*60)
        print("FILE-BASED GRIDDING TEST SUITE")
        print("="*60)
        print(f"Executable: {self.executable}")
        print(f"Test directory: {self.test_dir}")
        print("")

        tests = [
            ("Cluster Centers", self.test_cluster_centers),
            ("Random Points", self.test_random_points),
            ("Boundary Points", self.test_boundary_points),
        ]

        results = {}
        for name, test_func in tests:
            try:
                results[name] = test_func()
            except Exception as e:
                print(f"\n✗ TEST '{name}' CRASHED: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {status}: {name}")

        n_passed = sum(results.values())
        n_total = len(results)

        print("")
        print(f"Total: {n_passed}/{n_total} tests passed")
        print("="*60)

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Test suite for file-based gridding functionality"
    )
    parser.add_argument(
        "executable",
        help="Path to parent_gridder executable"
    )
    parser.add_argument(
        "--test-dir",
        default="tests/data",
        help="Directory for test files (default: tests/data)"
    )

    args = parser.parse_args()

    # Check executable exists
    if not os.path.exists(args.executable):
        print(f"ERROR: Executable not found: {args.executable}")
        return 1

    # Run tests
    tester = FileGriddingTest(args.executable, args.test_dir)
    success = tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
