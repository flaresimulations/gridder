import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Get a 3D mass histogram of a SWIFT simulation."
)
parser.add_argument(
    "--grid",
    type=str,
    help="The grid file to plot",
)
parser.add_argument(
    "--output",
    type=str,
    help="The file path for the plot",
)
parser.add_argument(
    "--smoothed",
    type=float,
    help="Are we plotting a mass grid or smoothed overdensity?",
    default=False,
)
parser.add_argument(
    "--zwidth",
    type=float,
    help="The distance to project along the z axis in Mpc.",
    default=25,
)
args = parser.parse_args()

# Get the commandline arguments
filepath = args.grid
outpath = args.output
smoothed = args.smoothed
zwidth = args.zwidth

# Open file
hdf = h5py.File(filepath, "r")

if smoothed:
    print("Not done yet")
else:
    parent = hdf["Parent"]
    boxsize = parent.attrs["Boxsize"]

    grid_cell_width = hdf["Grid"].attrs["CellWidth"]
    grid_cell_vol = grid_cell_width**3

    # Compute actual kernel width
    cells_per_kernel = np.int32(np.ceil(zwidth / grid_cell_width[0]))
    kernel_width = cells_per_kernel * grid_cell_width

    # Get grid
    grid = np.sum(hdf["MassGrid"][:, :, 0:cells_per_kernel], axis=-1)
    log_grid = np.zeros(grid.shape)
    log_grid[grid > 0] = np.log10(grid[grid > 0])

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

im = ax.imshow(grid, cmap="viridis", extent=[0, boxsize[0], 0, boxsize[1]])

cbar = fig.colorbar(im)
cbar.set_label("$(1 + \delta)$")

ax.set_xlabel("$x / [\mathrm{cMpc}]$")
ax.set_ylabel("$y / [\mathrm{cMpc}]$")

fig.savefig(outpath, bbox_inches="tight")

plt.close()

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

im = ax.imshow(log_grid, cmap="viridis", extent=[0, boxsize[0], 0, boxsize[1]])

cbar = fig.colorbar(im)
cbar.set_label("$\log_{10}(1 + \delta)$")

ax.set_xlabel("$x / [\mathrm{cMpc}]$")
ax.set_ylabel("$y / [\mathrm{cMpc}]$")

fig.savefig(
    f"{'/'.join(outpath.split('.')[-1])}_log.png",
    bbox_inches="tight",
)

plt.close()
