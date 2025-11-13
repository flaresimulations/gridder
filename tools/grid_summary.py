#!/usr/bin/env python3
"""
Grid Summary Tool

Analyzes overdensity statistics from gridded HDF5 files produced by the parent_gridder.

Usage:
    python tools/grid_summary.py <path_to_grid_file.hdf5>
"""

import sys
import h5py
import numpy as np
from scipy.spatial.distance import pdist


def analyze_grid_file(filename):
    """Analyze overdensity statistics from a grid HDF5 file."""
    
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Analyzing grid file: {filename}")
            print("=" * 60)
            
            # Find all kernel datasets
            grids_group = f['/Grids']
            kernel_datasets = []
            
            for key in grids_group.keys():
                if key.startswith('Kernel_'):
                    kernel_radius = key.replace('Kernel_', '')
                    kernel_datasets.append((kernel_radius, key))
            
            kernel_datasets.sort(key=lambda x: float(x[0]))
            
            if not kernel_datasets:
                print("No kernel datasets found in the file.")
                return
            
            # Analyze each kernel
            for kernel_radius, kernel_key in kernel_datasets:
                overdensities = f[f'/Grids/{kernel_key}/GridPointOverDensities'][:]
                
                print(f"\nKernel radius: {kernel_radius}")
                print("-" * 40)
                
                print(f"Grid points: {len(overdensities)}")
                print(f"Overdensity statistics:")
                print(f"  Min:    {np.min(overdensities):9.6f}")
                print(f"  Max:    {np.max(overdensities):9.6f}")
                print(f"  Mean:   {np.mean(overdensities):9.6f}")
                print(f"  Median: {np.median(overdensities):9.6f}")
                print(f"  Std:    {np.std(overdensities):9.6f}")
                
                # Count positive and negative values
                positive = np.sum(overdensities > 0)
                negative = np.sum(overdensities < 0)
                zero = np.sum(overdensities == 0)
                
                print(f"\nValue distribution:")
                print(f"  Positive (overdense): {positive:4d} ({positive/len(overdensities)*100:5.1f}%)")
                print(f"  Negative (underdense): {negative:4d} ({negative/len(overdensities)*100:5.1f}%)")
                print(f"  Zero:                 {zero:4d} ({zero/len(overdensities)*100:5.1f}%)")
                
                # Histogram
                bins = [-1, -0.5, 0, 0.5, 1, 2, 5, 10]
                if np.max(overdensities) > 10:
                    bins.append(np.max(overdensities))
                
                hist, _ = np.histogram(overdensities, bins=bins)
                print(f"\nHistogram:")
                for i in range(len(bins)-1):
                    print(f"  {bins[i]:6.1f} to {bins[i+1]:6.1f}: {hist[i]:4d} points")
                
                # Extreme values
                if len(overdensities) > 0:
                    min_idx = np.argmin(overdensities)
                    max_idx = np.argmax(overdensities)
                    print(f"\nExtreme values:")
                    print(f"  Most underdense:  δ = {overdensities[min_idx]:8.6f} (grid point {min_idx})")
                    print(f"  Most overdense:   δ = {overdensities[max_idx]:8.6f} (grid point {max_idx})")
            
            # Show grid structure info if available
            if '/Grids/GridPointPositions' in f:
                positions = f['/Grids/GridPointPositions'][:]
                print(f"\nGrid geometry:")
                print(f"  Total grid points: {len(positions)}")
                print(f"  Position ranges:")
                print(f"    X: {np.min(positions[:, 0]):8.3f} to {np.max(positions[:, 0]):8.3f}")
                print(f"    Y: {np.min(positions[:, 1]):8.3f} to {np.max(positions[:, 1]):8.3f}")
                print(f"    Z: {np.min(positions[:, 2]):8.3f} to {np.max(positions[:, 2]):8.3f}")
                
                # Estimate grid spacing
                if len(positions) > 1:
                    # For uniform grids, find minimum non-zero distance
                    distances = pdist(positions[:min(100, len(positions))])  # Sample for speed
                    min_dist = np.min(distances[distances > 0])
                    print(f"  Estimated grid spacing: {min_dist:.3f}")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Expected dataset not found in file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python tools/grid_summary.py <path_to_grid_file.hdf5>")
        sys.exit(1)
    
    filename = sys.argv[1]
    analyze_grid_file(filename)


if __name__ == "__main__":
    main()