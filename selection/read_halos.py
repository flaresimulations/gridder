import h5py
import numpy as np

from read_grid import read_grid

"""
Read in the halo properties
"""
soap_file = ('/cosma8/data/dp004/flamingo/Runs/L2800N5040'
             '/DMO_FIDUCIAL/SOAP/halo_properties_0078.hdf5')

with h5py.File(soap_file, 'r') as hf:
    mhalo = hf['SO/200_crit/TotalMass'][:]
    com = hf['SO/200_crit/CentreOfMass'][:]


"""
Read in the grid
"""
overd, coods = read_grid()


# Select an overdensity, read in haloes 
# associated with it

idx = 0

coods[0]

