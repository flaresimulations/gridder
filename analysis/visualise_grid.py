import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from flare import plt as flareplt

# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Get the simulation "tag"
sim_tag = sys.argv[2]

# Get the simulation "type"
sim_type = sys.argv[3]

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]
snap = snaps[num]

# Define path to file
metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
       "overdensity_gridding/" + sim_tag + "/" \
       + sim_type + "/snap_" + snap + "/" + metafile

# Open file
hdf = h5py.File(path, "r")

parent = hdf["Parent"]
boxsize = parent.attrs["Boxsize"]

grid_cell_width = hdf["Delta_grid"].attrs["Cell_Width"]
grid_cell_vol = grid_cell_width ** 3

# Compute actual kernel width
cells_per_kernel = np.int32(np.ceil(25 / grid_cell_width[0]))
kernel_width = cells_per_kernel * grid_cell_width

# Get grid
grid = np.sum(hdf["Parent_Grid"][:, :, 0:cells_per_kernel], axis=-1)
log_grid = np.zeros(grid.shape)
log_grid[grid > 0] = np.log10(grid[grid > 0])

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

im = ax.imshow(grid, cmap="viridis", extent=[0, boxsize[0], 0, boxsize[1]])

cbar = fig.colorbar(im)
cbar.set_label("$(1 + \delta)$")

ax.set_xlabel("$x / [\mathrm{cMpc}]$")
ax.set_ylabel("$y / [\mathrm{cMpc}]$")

fig.savefig("plots/overdensity_gird_" + sim_tag + "_" + sim_type + "_" + snap + ".png",
            bbox_inches="tight")

plt.close()

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

im = ax.imshow(log_grid, cmap="viridis", extent=[0, boxsize[0], 0, boxsize[1]])

cbar = fig.colorbar(im)
cbar.set_label("$\log_{10}(1 + \delta)$")

ax.set_xlabel("$x / [\mathrm{cMpc}]$")
ax.set_ylabel("$y / [\mathrm{cMpc}]$")

fig.savefig("plots/log_overdensity_loggrid_" + sim_tag + "_" + sim_type + "_" + snap + ".png",
            bbox_inches="tight")

plt.close()
