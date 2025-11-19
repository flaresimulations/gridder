# Gridding

Three methods for generating grid points:

| Type | Description | Use Case |
|------|-------------|----------|
| **uniform** | Regular cubic lattice | Density maps, uniform sampling |
| **random** | Monte Carlo sampling | Statistics, avoiding grid artifacts |
| **file** | Custom coordinates | Targeted regions, halos |

## Uniform Grids

Regular cubic lattice of `cdim³` points.

```yaml
Grid:
  type: uniform
  cdim: 100  # Creates 1,000,000 points
```

Points placed at: $(i+0.5)/N \times L_{\rm box}$ for $i \in [0, N-1]$

**Memory:** ~80 MB per million points (5 kernels)

**Resolution selection:**
- Nyquist: `cdim ≥ L_box / R` (where R = kernel radius)
- Recommended: 2-3× Nyquist

**Use for:** Density maps, FFT processing, uniform sampling

---

## Random Grids

Randomly distributed points.

```yaml
Grid:
  type: random
  n_grid_points: 1000000
```

Points uniformly sampled from simulation volume using Mersenne Twister (seed=0, reproducible).

**Average spacing:** $(V / N)^{1/3}$

**Use for:** Statistics (PDFs, correlations), avoiding grid artifacts

---

## File-Based Grids

Load coordinates from text file.

```yaml
Grid:
  type: file
  grid_file: /path/to/grid_points.txt
```

**File format:**
```
# x y z (one point per line, comoving Mpc/h)
10.5 20.3 15.7
25.0 30.0 35.0
```

**Placeholder support:**
```yaml
grid_file: /data/grids/grid_points_0000.txt  # Replaced with snapshot number
```

### Examples

**Halo centers:**
```python
# Extract halo positions from catalog
with h5py.File('halos.hdf5', 'r') as f:
    pos = f['Halos/Coordinates'][:]
    np.savetxt('halo_centers.txt', pos, fmt='%.6f')
```

**Use for:** Targeted analysis of specific structures

## See Also

- **[Parameters](parameters.md)** - Grid parameter reference
- **[Quickstart](quickstart.md)** - Basic examples
- **[Runtime Arguments](runtime-arguments.md)** - Command line options
