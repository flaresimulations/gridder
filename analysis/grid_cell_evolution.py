import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib as mpl
from flare import plt as flareplt
import h5py
import sys
plt.rcParams['axes.grid'] = True


# Define the snapshot strings
snaps = [str(i).zfill(4) for i in range(19, -1, -1)]

slopes = {}
odens = {}
zs = {}

# Set up redshift norm
norm = cm.Normalize(vmin=2, vmax=15)
cmap = plt.get_cmap('plasma', len(snaps))

# Loop over snapshots
prev_grid = None
prev_time = None
for snap in snaps:

    print(snap)

    # Define path to file
    metafile = "overdensity_L2800N5040_DMO_snap%s.hdf5" % snap
    path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
        "overdensity_gridding/L2800N5040/DMO/snap_" + snap + "/" + metafile

    # Open file
    hdf = h5py.File(path, "r")

    mean_density = hdf["Parent"].attrs["Mean_Density"]
    z = hdf["Parent"].attrs["Redshift"]
    grid = (hdf["Parent_Grid"][...] / mean_density) - 1

    hdf.close()

    # Compute the logarithmic slope of overdensity
    if prev_grid is not None:

        delta = grid / prev_grid

        slopes[snap] = delta.flatten()
        odens[snap] = np.log10(grid.flatten() + 1)
        zs[snap] = z

    prev_grid = grid


fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for snap in zs.keys():

    print("Plotting hist: %s" % snap)

    H, bins = np.histogram(slopes[snap][~np.isnan(slopes[snap])], bins=50)
    bin_cents = (bins[1:] + bins[:-1]) / 2

    ax.plot(bin_cents, H, color=cmap(norm(zs[snap])), alpha=0.7)

ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm)
cb1.set_label("$z$")

ax.set_xlabel("$\Delta_B / \Delta_A$")
ax.set_ylabel("$N$")

fig.savefig("plots/delta_overdensity_z_L2800N5040_DMO.png",
            bbox_inches="tight")
plt.close(fig)

for snap in zs.keys():

    print("Plotting scatter: %s" % snap)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mappable = ax.hexbin(odens[snap][~np.isnan(slopes[snap])],
                         slopes[snap][~np.isnan(slopes[snap])],
                         mincnt=1,
                         linewidth=0.2, cmap="viridis")

    cb1 = fig.colorbar(mappable)
    cb1.set_label("$N$")

    ax.set_ylabel("$\Delta_B / \Delta_A$")
    ax.set_xlabel("$\log_{10}(\Delta_B + 1)$")

    fig.savefig("plots/delta_overdensity_hex_L2800N5040_DMO_%s.png" % snap,
                bbox_inches="tight")
    plt.close(fig)
