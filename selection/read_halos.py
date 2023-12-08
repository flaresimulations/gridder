import h5py
import numpy as np

import matplotlib.pyplot as plt

from read_grid import read_grid


snap = '0019' 
"""
Read in the halo properties
"""
soap_file = (f'/cosma8/data/dp004/flamingo/Runs/L2800N5040'
             f'/DMO_FIDUCIAL/SOAP/halo_properties_{snap}.hdf5')

with h5py.File(soap_file, 'r') as hf:
    scale_factor = hf['SWIFT/Cosmology'].attrs['Scale-factor']
    mhalo = hf['SO/200_crit/TotalMass'][:]
    com = hf['SO/200_crit/CentreOfMass'][:] * (1. / scale_factor)


"""
Read in the grid
"""
overd, coods = read_grid(snap=snap)


# Select an overdensity, read in haloes 
# associated with it

def calc_df(_x, volume, massBinLimits):
    hist, dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) /\
                (massBinLimits[1] - massBinLimits[0]) # Poisson errors

    return phi, phi_sigma, hist


volume_parent = 2800**3
phi_parent, _, _ = calc_df(np.log10(mhalo), volume_parent, binLimits)




idxs = []
idxs.append(np.random.choice(np.where(overd > 3)[0], size=2))
idxs.append(np.random.choice(np.where((overd < 3) & (overd > 2))[0], size=2))
idxs.append(np.random.choice(np.where((overd < 2) & (overd > 1))[0], size=2))
idxs.append(np.random.choice(np.where((overd < 1) & (overd > 0))[0], size=2))

fig, ax = plt.subplots(1, 1)

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=overd.min(), vmax=overd.max())

dx = 12.5
volume = (2*dx)**3  # 25 Mpc cube
binLimits = np.linspace(10,15,25)
bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2

for idx in np.hstack(idxs):
    halo_mask = (com[:,0] < (coods[idx,0] + dx)) &\
                (com[:,0] > (coods[idx,0] - dx)) &\
                (com[:,1] < (coods[idx,1] + dx)) &\
                (com[:,1] > (coods[idx,1] - dx)) &\
                (com[:,2] < (coods[idx,2] + dx)) &\
                (com[:,2] > (coods[idx,2] - dx))

    phi, _, _ = calc_df(np.log10(mhalo[halo_mask]), volume, binLimits)
    print(idx, overd[idx], phi)
    ax.step(bins, np.log10(phi), label=idx,
            color=cmap(norm(overd[idx])))


ax.step(bins, np.log10(phi_parent), label='Parent', color='red')
ax.legend()
plt.show()


