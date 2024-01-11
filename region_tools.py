"""A collection of tools for working with regions and region grid files.

Example Usage:
    grid_points = get_grid_points("/path/to/grid.hdf5", np.array([0, 0, 0]))
    grid_points = get_grid_points(
        "/path/to/grid.hdf5", np.array([[0, 0, 0], [1, 1, 1]])
    )

"""
import h5py
import numpy as np


def _get_single_grid_point(filepath, ix, iy, iz, pos):
    """
    Get a single grid point from the grid file.

    Args:
        filepath (str):
            The path to the grid file.
        ix (int):
            The x index of the grid point.
        iy (int):
            The y index of the grid point.
        iz (int):
            The z index of the grid point.
        pos (np.ndarray):
            The position of the grid point.
    """
    # Open the grid file
    with h5py.File(filepath, "r") as hdf:
        grid = hdf["CellGrids"][f"{ix}_{iy}_{iz}"]["OverDensity"]
        cell_edge = hdf["CellGrids"][f"{ix}_{iy}_{iz}"]["CellEdge"]

        # Get the grid position relative to the cell edge
        pos -= cell_edge

        # Get the grid point index
        x_index = int(np.floor(pos[0] / hdf["Grid"].attrs["CellWidth"][0]))
        y_index = int(np.floor(pos[1] / hdf["Grid"].attrs["CellWidth"][1]))
        z_index = int(np.floor(pos[2] / hdf["Grid"].attrs["CellWidth"][2]))

        return grid[x_index, y_index, z_index]


def get_grid_points(filepath, pos):
    """
    Get the grid point at the given coordinates.

    Args:
        filepath (str):
            The path to the grid file.
        pos (np.ndarray):
            The position of the grid point/s.
    """
    # Open the grid file
    with h5py.File(filepath, "r") as hdf:
        sim_width = hdf["Parent"].attrs["CellWidth"]

    # Split coordinates into x, y, z
    x = pos[:, 0] if pos.ndim == 2 else pos[0]
    y = pos[:, 1] if pos.ndim == 2 else pos[1]
    z = pos[:, 2] if pos.ndim == 2 else pos[2]

    # Get the sim cell for this position
    x_index = int(np.floor(x / sim_width[0]))
    y_index = int(np.floor(y / sim_width[1]))
    z_index = int(np.floor(z / sim_width[2]))

    # For multiple positions we need to do each individually
    if pos.ndim == 2:
        point_ovden = np.zeros((len(x), 3))
        for ind, (ix, iy, iz) in enumerate(zip(x_index, y_index, z_index)):
            # Open the file and extract the grid point relative to the edge
            # of the cell
            with h5py.File(filepath, "r") as hdf:
                point_ovden[ind] = _get_single_grid_point(
                    filepath, ix, iy, iz, pos[ind]
                )

        return point_ovden

    return _get_single_grid_point(filepath, x_index, y_index, z_index, pos)
