#!/usr/bin/env python3
"""
Analyze debug output to diagnose tree traversal bugs
"""
import h5py
import numpy as np
import sys

def analyze_file(filename):
    print('='*70)
    print(f'ANALYZING: {filename}')
    print('='*70)

    with h5py.File(filename, 'r') as f:
        # Get grid point positions
        positions = np.array(f['Grids/GridPointPositions'])

        print(f'\nGrid point locations:')
        for i, pos in enumerate(positions):
            print(f'  GP {i:2d}: ({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f})')

        print('\n' + '='*70)
        print('Per-kernel comparison:')
        print('='*70)

        for kernel_name in sorted(f['Grids'].keys()):
            if kernel_name == 'GridPointPositions':
                continue

            kernel_rad = f[f'Grids/{kernel_name}'].attrs['KernelRadius']
            octree_counts = np.array(f[f'Grids/{kernel_name}/GridPointCounts'])
            brute_counts = np.array(f[f'Grids/{kernel_name}/BruteForceGridPointCounts'])

            print(f'\n{kernel_name} (radius = {kernel_rad} Mpc/h):')
            print(f'  GP  | Position              | Octree | Brute  | Diff   | Status')
            print(f'  ----+----------------------+--------+--------+--------+-------')

            for i in range(len(octree_counts)):
                pos_str = f'({positions[i][0]:5.1f},{positions[i][1]:5.1f},{positions[i][2]:5.1f})'
                diff = octree_counts[i] - brute_counts[i]

                if diff == 0:
                    status = 'OK'
                elif octree_counts[i] == 0 and brute_counts[i] > 0:
                    status = 'MISS'  # Octree missed particles
                elif octree_counts[i] > brute_counts[i]:
                    status = 'OVER'  # Octree over-counted
                else:
                    status = 'UNDER' # Octree under-counted

                print(f'  {i:3d} | {pos_str:20s} | {octree_counts[i]:6d} | {brute_counts[i]:6d} | {diff:+7d} | {status}')

            # Summary
            total_mismatches = np.sum(octree_counts != brute_counts)
            total_miss = np.sum((octree_counts == 0) & (brute_counts > 0))
            total_over = np.sum(octree_counts > brute_counts)

            print(f'\n  Summary: {total_mismatches}/{len(octree_counts)} mismatches')
            print(f'    - Complete misses (0 when should find particles): {total_miss}')
            print(f'    - Over-counting (found more than truth): {total_over}')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_file(sys.argv[1])
    else:
        # Default files to analyze
        files = [
            'tests/data/boundary_output.hdf5',
            'tests/data/cluster_output.hdf5',
        ]
        for f in files:
            try:
                analyze_file(f)
                print('\n')
            except FileNotFoundError:
                print(f'File not found: {f}\n')
