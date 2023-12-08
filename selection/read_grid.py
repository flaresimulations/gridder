import h5py

def read_grid(fname=('/cosma7/data/dp004/FLARES/FLARES-2/'
                     'Parent/overdensity_gridding/L2800N5040/DMO'),
              snap='0019'):

    grid_fname = (f'{fname}/snap_{snap}/'
                  f'smoothed_overdensity_L2800N5040_DMO_snap{snap}_kernel25.hdf5')
                  # f'smoothed_overdensity_L2800N5040_DMO_snap{snap}_kernel25_rank0.hdf5')

    with h5py.File(grid_fname, 'r') as hf:
        overd = hf['Region_Overdensity'][:]
        coods = hf['Region_Centres'][:]
        # hf['Region_Overdensity_Stdev'][:]
        # hf['Region_Indices'][:]

    return overd, coods
