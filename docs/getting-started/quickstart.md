# Quick Start

Get up and running with the FLARES-2 Gridder in minutes.

## Prerequisites

Ensure you have [installed](installation.md) the gridder and all dependencies.

## Step 1: Create a Parameter File

Create `example_params.yml`:

```yaml
Kernels:
  nkernels: 3
  kernel_radius_1: 0.5
  kernel_radius_2: 1.0
  kernel_radius_3: 2.0

Grid:
  type: uniform
  cdim: 50

Tree:
  max_leaf_count: 200

Input:
  filepath: /path/to/snapshot.hdf5

Output:
  filepath: ./output/
  basename: gridded_data.hdf5
  write_masses: 1
