import sys

import h5py
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import flare.plt as fplt
import numpy as np

plt.rcParams['axes.grid'] = True

# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Get the simulation "tag"
sim_tag = sys.argv[2]

# Get the simulation "type"
sim_type = sys.argv[3]

# Define kernel width
ini_kernel_width = int(sys.argv[4])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]
snap = snaps[num]

# Define output paths
metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
         "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

# Define path to file
file = "smoothed_" + metafile.split(".")[
    0] + "_kernel%d.hdf5" % ini_kernel_width
path = outdir + file

# Open file
hdf = h5py.File(path, "r")

z = hdf.attrs["Parent_Redshift"]
grid = hdf["Region_Overdensity"][...]
grid_std = hdf["Region_Overdensity_Stdev"][...]

hdf.close()

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.hexbin(np.log10(grid), grid_std,
               norm=cm.LogNorm(),
               gridsize=100, mincnt=1,
               linewidths=0.2, cmap='viridis')

ax.set_xlabel("$\log_{10}(1 + \delta)$")
ax.set_ylabel("$\sigma(1 + \delta)$")

ax.text(0.95, 0.05, f'$z={z}$',
        bbox=dict(boxstyle="round,pad=0.3", fc='w',
                  ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment='right')

cbar = fig.colorbar(im)
cbar.set_label("$N$")

fig.savefig(
    "plots/region_spread_kernel" + str(ini_kernel_width) + "_" + snap + ".png",
    bbox_inches="tight")

plt.close()
