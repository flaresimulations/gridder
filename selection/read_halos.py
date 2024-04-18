import numpy as np

import matplotlib.pyplot as plt

from methods import read_grid, read_haloes, calc_df


binLimits = np.linspace(10, 15, 25)
bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1]) / 2

snap = "0019"
# directory = "/snap8/scratch/dp004/dc-rope1/FLARES-2/gridded_data/FLAMINGO/L5600N5040/DMO_FIDUCIAL/"

# Read in the grid
overd, coods = read_grid(snap=snap)

# Read in haloes
mhalo, com = read_haloes(snap=snap)

# Select an overdensity, read in haloes associated with it
idxs = []
idxs.append(np.random.choice(np.where(overd > 3)[0], size=2))
idxs.append(np.random.choice(np.where((overd < 3) & (overd > 2))[0], size=2))
idxs.append(np.random.choice(np.where((overd < 2) & (overd > 1))[0], size=2))
idxs.append(np.random.choice(np.where((overd < 1) & (overd > 0))[0], size=2))

# Calculate HMF of whole parent
volume_parent = 2800**3
phi_parent, _, _ = calc_df(np.log10(mhalo), volume_parent, binLimits)


fig, ax = plt.subplots(1, 1)

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=overd.min(), vmax=overd.max())

dx = 12.5
volume = (2 * dx) ** 3  # 25 Mpc cube

for idx in np.hstack(idxs):
    halo_mask = (
        (com[:, 0] < (coods[idx, 0] + dx))
        & (com[:, 0] > (coods[idx, 0] - dx))
        & (com[:, 1] < (coods[idx, 1] + dx))
        & (com[:, 1] > (coods[idx, 1] - dx))
        & (com[:, 2] < (coods[idx, 2] + dx))
        & (com[:, 2] > (coods[idx, 2] - dx))
    )

    phi, _, _ = calc_df(np.log10(mhalo[halo_mask]), volume, binLimits)
    # print(idx, overd[idx], phi)
    ax.step(bins, np.log10(phi), label=idx, color=cmap(norm(overd[idx])))


ax.step(bins, np.log10(phi_parent), label="Parent", color="red")
ax.legend()
plt.show()
