import h5py
import numpy as np


def read_grid(
    fname=(
        "/snap8/scratch/dp004/dc-rope1/FLARES-2/"
        "gridded_data/FLAMINGO/L5600N5040/DMO_FIDUCIAL/"
    ),
    snap="0019",
    kernel="10p0",
):

    grid_fname = f"{fname}/grid_{snap}_kernel{kernel}.hdf5"

    with h5py.File(grid_fname, "r") as hf:
        cDim = hf['Parent'].attrs['CDim']  # 

        overd = hf["Region_Overdensity"][:]
        coods = hf["Region_Centres"][:]
        # hf['Region_Overdensity_Stdev'][:]
        # hf['Region_Indices'][:]

    return overd, coods


def read_haloes(
    directory=(
        "/cosma8/data/dp004/flamingo/Runs/L2800N5040/DMO_FIDUCIAL/SOAP/"
    ),
    snap="0019",
):
    """
    Read in the halo properties
    """
    soap_file = f"{directory}/halo_properties_{snap}.hdf5"

    with h5py.File(soap_file, "r") as hf:
        scale_factor = hf["SWIFT/Cosmology"].attrs["Scale-factor"]
        mhalo = hf["SO/200_crit/TotalMass"][:]
        com = hf["SO/200_crit/CentreOfMass"][:] * (1.0 / scale_factor)

    return mhalo, com


def calc_df(_x, volume, massBinLimits):
    hist, dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) / (
        massBinLimits[1] - massBinLimits[0]
    )  # Poisson errors

    return phi, phi_sigma, hist
