import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib as mpl
from flare import plt as flareplt
from schwimmbad import MultiPool
import h5py
import sys
plt.rcParams['axes.grid'] = True


def distributed_calc(snap):

    # Define the previous snapshot
    prev_snap = str(int(snap) - 1).zfill(4)

    print(prev_snap, "->", snap)

    # Define path to file
    metafile = "smoothed_overdensity_L2800N5040_DMO_snap%s_kernel25.hdf5" % snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
        "overdensity_gridding/L2800N5040/DMO/snap_" + snap + "/" + metafile

    # Open file
    hdf = h5py.File(path, "r")

    z = hdf.attrs["Parent_Redshift"]
    grid = hdf["Region_Overdensity"][...]

    hdf.close()

    # Define path to file
    metafile = "smoothed_overdensity_L2800N5040_DMO_snap%s_kernel25.hdf5" % prev_snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
        "overdensity_gridding/L2800N5040/DMO/snap_" + prev_snap + "/" + metafile

    # Open file
    hdf = h5py.File(path, "r")

    prev_z = hdf.attrs["Parent_Redshift"]
    prev_grid = hdf["Region_Overdensity"][...]

    hdf.close()

    delta = grid - prev_grid
    
    slopes = delta.flatten()
    odens = np.log10(grid.flatten())

    # Plot the distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mappable = ax.hexbin(odens,
                         slopes,
                         mincnt=1, norm=cm.LogNorm(),
                         linewidth=0.2, cmap="viridis")

    cb1 = fig.colorbar(mappable)
    cb1.set_label("$N$")

    ax.set_ylabel("$\delta_B - \delta_A$")
    ax.set_xlabel("$\log_{10}(1 + \delta_B)$")

    fig.savefig(
        "plots/smoothed25_delta_overdensity_hex_L2800N5040_DMO_%s.png" % snap,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Compute histogram
    H, bins = np.histogram(slopes, bins=50)
    bin_cents = (bins[1:] + bins[:-1]) / 2

    return z, snap, slopes, odens, H, bin_cents

# Define the snapshot strings
snaps = [str(i).zfill(4) for i in range(1, 20)]

slopes = {}
odens = {}
zs = {}
Hs = {}
xs = {}

# Set up redshift norm
norm = cm.Normalize(vmin=2, vmax=15)
cmap = plt.get_cmap('plasma', len(snaps))

# Compute the deltas
with MultiPool(int(sys.argv[1])) as pool:
    results = list(pool.map(distributed_calc, snaps))

for res in results:

    z, snap, slope, oden, H, bin_cents = res
    slopes[snap] = slopes
    odens[snap] = oden
    zs[snap] = z
    Hs[snap] = H
    xs[snap] = bin_cents

fig = plt.figure()
ax = fig.add_subplot(111)

for snap in zs.keys():

    print("Plotting hist: %s" % snap)

    ax.plot(xs[snap], Hs[snap], color=cmap(norm(zs[snap])), alpha=0.7)

ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm)
cb1.set_label("$z$")

ax.set_xlabel("$\Delta_B - \Delta_A$")
ax.set_ylabel("$N$")

fig.savefig("plots/smoothed25_delta_overdensity_z_L2800N5040_DMO.png",
            bbox_inches="tight")
plt.close(fig)
