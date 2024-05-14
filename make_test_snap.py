"""A script for generating a uniform snapshots.

This will generate a grid of particles in a box and write them to a
SWIFT snapshot file including the cell structure.
"""
import argparse
from tqdm import tqdm
import h5py
import numpy as np


def get_cell_index(x, y, z, cell_size, cdim):
    """Get the cell index for a particle at position (x, y, z).

    Parameters
    ----------
    x : float
        The x-coordinate of the particle.
    y : float
        The y-coordinate of the particle.
    z : float
        The z-coordinate of the particle.
    cell_size : float
        The size of a cell.
    cdim : int
        The number of cells in each dimension.

    Returns
    -------
    int
        The cell index.
    """
    i = int(x / cell_size)
    j = int(y / cell_size)
    k = int(z / cell_size)
    return k + j * cdim + i * cdim * cdim


def make_ics(filepath, cdim, gdim, boxsize, doner_path):
    """Create a snapshot with a uniform grid of particles.

    Parameters
    ----------
    filepath : str
        The path to the output file.
    cdim : int
        The number of cells in each dimension.
    grid_sep : float
        The number of grid particles in each dimension.
    boxsize : float
        The size of the box.
    """
    # Get the number of grid points from the boxsize and grid separation
    grid_sep = boxsize / gdim

    print(
        f"Creating a {gdim}x{gdim}x{gdim} grid of "
        f"particles with a separation of {grid_sep}."
    )

    # Create the grid of particles
    pos = np.zeros((gdim**3, 3))
    ind = 0
    for i in range(gdim):
        for j in range(gdim):
            for k in range(gdim):
                pos[ind, 0] = (i + 0.5) * grid_sep
                pos[ind, 1] = (j + 0.5) * grid_sep
                pos[ind, 2] = (k + 0.5) * grid_sep
                ind += 1

    # Assign each particle to a cell keeping track of the number of particles
    # in each cell
    cell_size = boxsize / cdim
    cell_count = np.zeros(cdim**3, dtype=np.int32)
    cells = {k: [] for k in range(gdim**3)}
    for i, (x, y, z) in enumerate(tqdm(pos)):
        cell = get_cell_index(x, y, z, cell_size, cdim)
        cell_count[cell] += 1
        cells[cell].append(i)

    # Convert counts into offsets
    cell_offset = np.cumsum(cell_count)
    cell_offset = np.insert(cell_offset, 0, 0)

    print("Number of particles in each cell:")
    print(cell_count)

    # Sort the particles by cell
    sinds = np.zeros(pos.shape[0], dtype=np.int32)
    for i in tqdm(range(cdim)):
        for j in range(cdim):
            for k in range(cdim):
                cell = k + j * cdim + i * cdim * cdim
                start = cell_offset[cell]
                end = start + cell_count[cell]
                sinds[start:end] = cells[cell]
    pos = pos[sinds, :]

    # Write the snapshot
    with h5py.File(filepath, "w") as hdf:
        # Write the particles
        part_type_1 = hdf.create_group("PartType1")
        part_type_1.create_dataset("Coordinates", data=pos)
        part_type_1.create_dataset("ParticleIDs", data=np.arange(pos.shape[0]))
        part_type_1.create_dataset("Masses", data=np.ones(pos.shape[0]))

        # Write the cell structure
        cell_struct = hdf.create_group("Cells")
        cell_struct.create_dataset("Counts/PartType1", data=cell_count)
        cell_struct.create_dataset("OffsetsInFile/PartType1", data=cell_offset)

        # Write the header
        header = hdf.create_group("Header")
        header.attrs["NumPart_ThisFile"] = np.array(
            [0, pos.shape[0], 0, 0, 0, 0]
        )
        header.attrs["NumPart_Total"] = np.array([0, pos.shape[0], 0, 0, 0, 0])
        header.attrs["NumPart_Total_HighWord"] = np.array([0, 0, 0, 0, 0, 0])
        header.attrs["MassTable"] = np.array([0, 1, 0, 0, 0, 0])
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = np.array([boxsize, boxsize, boxsize])

        # Write the cells metadata
        cell_meta = cell_struct.create_group("Meta-data")
        cell_meta.attrs["dimension"] = np.array(
            [cdim, cdim, cdim], dtype=np.int32
        )
        cell_meta.attrs["size"] = np.array([cell_size, cell_size, cell_size])

        # Get the units from the doner snapshot
        with h5py.File(doner_path, "r") as doner:
            units = doner["Units"]
            unit_mass = units.attrs["Unit mass in cgs (U_M)"]
            unit_length = units.attrs["Unit length in cgs (U_L)"]
            hdf.create_group("Units")
            hdf["Units"].attrs["Unit mass in cgs (U_M)"] = unit_mass
            hdf["Units"].attrs["Unit length in cgs (U_L)"] = unit_length

        # Get the cosmology we'll need from the doner snapshot
        with h5py.File(doner_path, "r") as donor:
            cosmology = donor["Cosmology"]
            mean_density = cosmology.attrs["Critical density [internal units]"]
            hdf.create_group("Cosmology")
            hdf["Cosmology"].attrs[
                "Critical density [internal units]"
            ] = mean_density


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        help="The path to the output file.",
    )
    parser.add_argument(
        "--cdim",
        type=int,
        help="The number of cells in each dimension.",
    )
    parser.add_argument(
        "--grid_dim",
        type=int,
        help="The number of grid cells along an axis to put a particle in.",
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        help="The size of the box.",
    )
    parser.add_argument(
        "--doner",
        type=str,
        help="The path to the donor snapshot.",
    )

    args = parser.parse_args()

    # Create the snapshot
    make_ics(args.output, args.cdim, args.grid_dim, args.boxsize, args.doner)
