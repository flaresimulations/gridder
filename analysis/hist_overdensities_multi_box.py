import numpy as np
import matplotlib.pyplot as plt
from flare import plt as flareplt
import h5py
import sys
plt.rcParams['axes.grid'] = True


# Get the commandline argument for which snapshot
num = int(sys.argv[1])

# Extract the snapshot string
snaps = [str(i).zfill(4) for i in range(0, 21)]
snap = snaps[num]

# Set up bins
step = 0.1
bin_edges = np.arange(0.00001, 15 + step, step)
bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2
step = 0.05
log_bin_edges = np.arange(-1.0, 3.0 + step, step)
log_bin_cents = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2

hists = {}

for sim_tag in ["L2800N5040", ]:
       for sim_type in ["HYDRO", "DMO"]:
           hists[sim_tag + "_" + sim_type] = {}

           # Define path to file
           metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
           path = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
                 "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/" + metafile

           # Open file
           hdf = h5py.File(path, "r")

           mean_density = hdf["Parent"].attrs["Mean_Density"]
           grid = hdf["Parent_Grid"][...]

           hdf.close()

           # Get counts for this cell
           hists[sim_tag + "_" + sim_type]["H"], _ = np.histogram(grid, bins=bin_edges)

           # Get counts for this cell
           hists[sim_tag + "_" + sim_type]["log_H"], _ = np.histogram(np.log10(grid), bins=log_bin_edges)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for sim_tag in ["L2800N5040", ]:
       for sim_type in ["HYDRO", "DMO"]:
              ax.plot(bin_cents, hists[sim_tag + "_" + sim_type]["H"],
                      label=sim_tag + "_" + sim_type + "_2Mpc")

ax.set_xlabel("$1 + \delta$")
ax.set_ylabel("$N$")

ax.legend()

fig.savefig("plots/overdensity_hist_boxes_" + snap + ".png",
            bbox_inches="tight")

plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for sim_tag in ["L2800N5040", ]:
       for sim_type in ["HYDRO", "DMO"]:
              ax.plot(log_bin_cents, hists[sim_tag + "_" + sim_type]["log_H"],
                      label=sim_tag + "_" + sim_type + "_2Mpc")

ax.set_xlabel("$\log_{10}(1 + \delta)$")
ax.set_ylabel("$N$")

ax.legend()

fig.savefig("plots/log_overdensity_hist_" + snap + ".png",
            bbox_inches="tight")

plt.close()
