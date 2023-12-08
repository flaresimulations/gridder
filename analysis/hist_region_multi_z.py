import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib as mpl
from flare import plt as flareplt
import h5py
import sys


plt.rcParams['axes.grid'] = True


# Get the simulation "tag"
sim_tag = sys.argv[1]

# Get the simulation "type"
sim_type = sys.argv[2]

# Define kernel width
ini_kernel_width = int(sys.argv[3])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 20)]

# Set up bins
step = 0.1
bin_edges = np.arange(0.00001, 15 + step, step)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2
step = 0.03
log_bin_edges = np.arange(-0.6, 0.6 + step, step)
log_bin_cents = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2

# Set up redshift norm
norm = cm.Normalize(vmin=2, vmax=15)
cmap = plt.get_cmap('plasma', len(snaps))

fig_log = plt.figure()
ax_log = fig_log.add_subplot(111)
ax_log.semilogy()

for num, snap in enumerate(reversed(snaps)):

       # Define output paths
       metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
       outdir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
                "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

       # Define path to file
       file = "smoothed_" + metafile.split(".")[0] + "_kernel%d.hdf5" % ini_kernel_width
       path = outdir + file

       # Open file
       hdf = h5py.File(path, "r")

       z = hdf.attrs["Parent_Redshift"]
       grid = hdf["Region_Overdensity"][...]

       print(snap, z)

       hdf.close()

       # Get counts for this cell
       log_H, _ = np.histogram(np.log10(grid), bins=log_bin_edges)

       # Plot this snapshot
       ax_log.plot(log_bin_cents, log_H, color=cmap(norm(z)))

ax_log.set_xlabel("$\log_{10}(1 + \delta)$")
ax_log.set_ylabel("$N$")

ax2_log = fig_log.add_axes([0.95, 0.1, 0.015, 0.8])
cb1 = mpl.colorbar.ColorbarBase(ax2_log, cmap=cmap,
                                norm=norm)
cb1.set_label("$z$")

fig_log.savefig("plots/log_region_multiz_" + str(ini_kernel_width) + "_"  + sim_tag + "_" + sim_type + ".png",
                bbox_inches="tight")
plt.close(fig_log)
