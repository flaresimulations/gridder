""" A module for smoothing grids over a spherical kernel.

This expects a file in the the output format produced by ParentGridder. It will
smooth every grid cell over an arbitrarily size spherical top hat kernel and
produce a distributed set of files followed by a single combined file.

Examples usage:

    smoother = GridSmoother(25, "grid.hdf5")
    smoother.decomp_cells()
    smoother.smooth_grid_cells()
    smoother.write_smoothed_grid_rankfile(outpath)

"""
import os
import h5py
import numpy as np
from mpi4py import MPI


class GridSmoother:
    """
    A class for smoothing grids over a spherical kernel.

    Attributes:
        grid_file (str):
            Path to the grid file.
        cell_width (float):
            Width of each grid cell.
        boxsize (float):
            Size of the simulation box.
        mean_density (float):
            Mean density of the simulation.
        ngrid_cells (int):
            Total number of grid cells in the simulation.
        cdim (tuple):
            Dimensions of the grid array.
        mass_gird (numpy.ndarray):
            Mass grid dataset.
        kernel_width (float):
            Width of the spherical top hat kernel.
        kernel_rad (float):
            Radius of the spherical top hat kernel.
        cells_in_kernel (int):
            Number of cells within the kernel.
        rank_cells (numpy.ndarray):
            Array representing the distribution of grid cells among ranks.
        rank_ncells (int):
            Number of cells assigned to the current rank.
        kernel_over_dens (numpy.ndarray):
            Array to store smoothed region values.
        kernel_stds (numpy.ndarray):
            Array to store smoothed region standard deviations.

    """

    def __init__(self, kernel_width, grid_file):
        """
        Set up the grid and kernel to be applied.

        Args:
            kernel_width (float):
                Width of the spherical top hat kernel.
            grid_file (str):
                Path to the grid file in the format produced by ParentGridder.
            rank (int):
                The rank ID of the current rank.

        """

        # Get the MPI information we need
        self.comm = None
        self.nranks = None
        self.rank = None
        self._setup_mpi()

        # Set up the grid
        self.grid_file = grid_file

        # Basic I/O information
        outpath = "smoothed_" + grid_file.split("/")[-1]
        self.out_basename = outpath.split(".")[0]
        self.out_ext = outpath.split(".")[-1]
        self.out_dir = "/".join(grid_file.split("/")[:-1]) + "/"

        # Set up variables to be populated when opening the grid
        self.cell_width = None
        self.boxsize = None
        self.mean_density = None
        self.ngrid_cells = None
        self.cdim = None

        # Read the grid attributes
        self._read_attrs()

        # The mass grid dataset
        self.mass_gird = None

        # Define kernel properties
        self.kernel_width = kernel_width
        self.kernel_rad = kernel_width / 2
        self.kernel_vol = 4 / 3 * np.pi * self.kernel_rad**3

        # How many cells are in a kernel?
        self.cells_in_kernel = int(np.ceil(self.kernel_width / self.cell_width[0])) + 1
        self.pad_cells = self.cells_in_kernel // 2 + 1

        # Define the decomposition attributes
        self.rank_cells = None
        self.rank_ncells = None
        self.x_ncells_rank = None

        # Set up the outputs to write out
        self.kernel_over_dens = None
        self.kernel_stds = None

    def _read_attrs(self):
        """
        Read the grid file for attributes.


        """

        # Open file
        hdf = h5py.File(self.grid_file, "r")

        # Get metadata
        self.boxsize = hdf["Parent"].attrs["Boxsize"]
        self.mean_density = hdf["Parent"].attrs["MeanDensity"]
        self.cdim = hdf["Grid"].attrs["CDim"]
        self.ngrid_cells = np.product(self.cdim)
        self.cell_width = hdf["Grid"].attrs["CellWidth"]
        self.cell_vol = hdf["Grid"].attrs["CellVolume"]

        # Print some nice things
        if self.rank == 0:
            print("Boxsize:", self.boxsize)
            print("Mean Density:", self.mean_density)
            print("Grid Cell Width:", self.cell_width)
            print("Grid CDim:", self.cdim)
            print("Grid cells total:", self.ngrid_cells)

        # Get results array dimensions
        self.cdim = hdf["MassGrid"].shape

        hdf.close()

    def _setup_mpi(self):
        """
        Sets up MPI communication.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

    def _apply_spherical_top_hat(self, subset, func, centre, cell_coords):
        """
        Applies a spherical top hat kernel to a cubic array and returns the sum
        of values inside the kernel.

        Args:
            subset (numpy.ndarray)
                The input cubic array.

        Returns:
            float
                The func applied to values inside the spherical top hat kernel.
        """

        # Create grid cell coordinates
        xx, yy, zz = np.meshgrid(
            cell_coords[0] - centre[0],
            cell_coords[1] - centre[1],
            cell_coords[2] - centre[2],
        )

        # Create a grid of coordinates centered at the middle of the array
        distances = np.sqrt(xx**2 + yy**2 + zz**2)

        # Create a boolean mask for the spherical top hat kernel
        mask = distances <= self.kernel_rad

        # Apply the mask to the input array and return the sum of values
        # inside the kernel
        return func(subset[mask])

    def get_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a grid cell.

        Args:
            cid (int)
                The flattened index.
        """
        i = int(cid / (self.cdim[1] * self.cdim[2]))
        j = int((cid / self.cdim[2]) % self.cdim[1])
        k = int(cid % self.cdim[2])

        return i, j, k

    def get_cellid(self, i, j, k):
        """
        Compute the flattened index of a grid/simulation cell coordinates.

        Args:
            i, j, k (int)
                The integer cell coordinates.
        """
        return k + self.cdim[2] * (j + self.cdim[1] * i)

    def domain_decomp(self):
        """
        Divide grid cells into Nranks slices along the i direction. We don't
        care about the exact weight of each slice since every operation costs
        the same amount.
        """

        assert self.nranks <= self.cdim[0], "Can't have more ranks than SWIFT cells."

        # Split the x direction amongst all ranks
        self.rank_cells = np.linspace(
            0,
            self.cdim[0],
            self.nranks + 1,
            dtype=int,
        )

        # How many x cells are on this rank?
        self.x_ncells_rank = self.rank_cells[self.rank + 1] - self.rank_cells[self.rank]

        # Loop over all cells and construct lists to be sliced
        self.my_cells = []
        for i in range(self.x_ncells_rank):
            for j in range(self.cdim[1]):
                for k in range(self.cdim[2]):
                    cell = self.get_cellid(i, j, k)
                    self.my_cells.append(cell)

        # Convert the list to an array
        self.my_cells = np.array(self.my_cells, dtype=int)

        # How many cells do we have?
        self.rank_ncells = self.my_cells.size

        print("Rank=", self.rank, "- My Ncells=", self.rank_ncells)

        # Open file to get the relevant grid cells
        hdf = h5py.File(self.grid_file, "r")

        # Get the mass grid (padding handled below)
        self.mass_grid = np.zeros(
            (
                self.x_ncells_rank + 2 * self.pad_cells,
                self.cdim[1] + 2 * self.pad_cells,
                self.cdim[2] + 2 * self.pad_cells,
            ),
            dtype=np.float32,
        )
        self.mass_grid[
            self.pad_cells : -self.pad_cells,
            self.pad_cells : -self.pad_cells,
            self.pad_cells : -self.pad_cells,
        ] = hdf["MassGrid"][
            self.rank_cells[self.rank] : self.rank_cells[self.rank + 1],
            :,
            :,
        ]

        # Add on the lower pad region handling wrapping
        if self.rank > 0:
            self.mass_grid[
                : self.pad_cells,
                self.pad_cells : -self.pad_cells,
                self.pad_cells : -self.pad_cells,
            ] = hdf["MassGrid"][
                self.rank_cells[self.rank]
                - self.pad_cells : self.rank_cells[self.rank],
                :,
                :,
            ]
        else:
            self.mass_grid[: self.pad_cells, :, :] = hdf["MassGrid"][
                -self.pad_cells :, :, :
            ]

        # Add on the upper pad region handling wrapping
        if self.rank < self.nranks - 1:
            self.mass_grid[
                -self.pad_cells :,
                self.pad_cells : -self.pad_cells,
                self.pad_cells : -self.pad_cells,
            ] = hdf["MassGrid"][
                self.rank_cells[self.rank + 1]
                - self.pad_cells : self.rank_cells[self.rank + 1],
                :,
                :,
            ]
        else:
            self.mass_grid[-self.pad_cells :, :, :] = hdf["MassGrid"][
                -self.pad_cells :, :, :
            ]

        # And finally pad the edges of the 2nd and 3rd axes for periodic
        # boundary conditions
        self.mass_grid[:, -self.pad_cells :, :] = self.mass_grid[:, : self.pad_cells, :]
        self.mass_grid[:, : self.pad_cells, :] = self.mass_grid[:, -self.pad_cells :, :]
        self.mass_grid[:, :, -self.pad_cells :] = self.mass_grid[:, :, : self.pad_cells]
        self.mass_grid[:, :, : self.pad_cells] = self.mass_grid[:, :, -self.pad_cells :]

        hdf.close()

    def smooth_grid_cells(self):
        """ """

        # Set up the outputs to write out
        self.kernel_over_dens = np.zeros(self.rank_ncells)
        self.kernel_stds = np.zeros(self.kernel_over_dens.size)

        # Define the x offset
        x_offset = self.rank_cells[self.rank]

        # Loop over this ranks cells
        for ind, cid in enumerate(self.my_cells):
            # Get the i j k coordinates
            i, j, k = self.get_cell_ijk(cid)

            # Calculate the centre relative to the grid
            centre = np.array(
                [
                    i - x_offset + self.pad_cells,
                    j + self.pad_cells,
                    k + self.pad_cells,
                ],
                dtype=int,
            )

            # Get the slice indices of the mass array we need to
            low_i = centre[0] - self.pad_cells
            low_j = centre[1] - self.pad_cells
            low_k = centre[2] - self.pad_cells

            # Get the mass grid
            sub_grid = self.mass_grid[
                low_i : low_i + self.cells_per_kernel + self.pad_cells,
                low_j : low_j + self.cells_per_kernel + self.pad_cells,
                low_k : low_k + self.cells_per_kernel + self.pad_cells,
            ]

            # Get the coordinates of each cell
            cell_coords = np.array(
                [
                    np.arange(low_i, low_i + self.cells_per_kernel + self.pad_cells)
                    + 0.5 * self.cell_width,
                    np.arange(low_j, low_j + self.cells_per_kernel + self.pad_cells)
                    + 0.5 * self.cell_width,
                    np.arange(low_k, low_k + self.cells_per_kernel + self.pad_cells)
                    + 0.5 * self.cell_width,
                ]
            )

            # And redefine the centre in simulation coordinates
            centre = (centre + 0.5) * self.cell_width

            # Apply the kernel and get the standard deviation in terms of
            # mass and log10(1 + delta) repectively (kernel sum is converted
            # to log10(1 + delta) below)
            kernel_sum = self._apply_spherical_top_hat(
                sub_grid, np.sum, centre, cell_coords
            )
            kernel_std = self._apply_spherical_top_hat(
                np.log10(sub_grid / self.cell_vol / self.mean_density),
                np.std,
                centre,
                cell_coords,
            )

            # Store the results (as overdensities) and index
            self.kernel_over_dens[ind] = np.log10(
                kernel_sum / self.kernel_vol / self.mean_density
            )
            self.kernel_stds[ind] = kernel_std

    def write_smoothed_grid_rankfile(self):
        """ """
        # Write out the results of smoothing
        rankfile = (
            f"{self.out_dir}{self.out_basename}_kernel"
            f"{int(self.kernel_width)}_rank{self.rank}.{self.out_ext}"
        )
        hdf = h5py.File(
            rankfile,
            "w",
        )
        hdf.attrs["KernelWidth"] = self.kernel_width
        hdf.create_dataset(
            "KernelOverdensity",
            data=self.kernel_over_dens,
            shape=self.kernel_over_dens.shape,
            dtype=self.kernel_over_dens.dtype,
            compression="gzip",
        )
        hdf.create_dataset(
            "KernelOverdensityStDev",
            data=self.kernel_stds,
            shape=self.kernel_stds.shape,
            dtype=self.kernel_stds.dtype,
            compression="gzip",
        )

        hdf_grid = h5py.File(self.grid_file, "r")

        for key in hdf_grid.attrs.keys():
            hdf.attrs[key] = hdf_grid.attrs[key]

        for root_key in ["Parent", "Grid", "Units"]:
            for key in hdf_grid[root_key].attrs.keys():
                hdf.attrs[root_key + "_" + key] = hdf_grid[root_key].attrs[key]

        hdf_grid.close()

        hdf.close()

    def combine_distributed_files(self, delete_distributed=False):
        """
        Creates a single output file containing all slices.

        This method should only be called by rank 0 but protections are in
        place incase this isn't the case.

        Args:
            delete_distributed (bool)
                Whether to remove the distributed files as they are processed.
                By default False.
        """

        # Early exit if were not on rank 0
        if self.rank != 0:
            return

        # Open the rank 0 distributed file to get attributes
        rank0file = (
            f"{self.out_dir}{self.out_basename}_kernel"
            f"{int(self.kernel_width)}_rank0.{self.out_ext}"
        )
        hdf_rank0 = h5py.File(rank0file, "r")

        # Create the single output file
        outpath = (
            f"{self.out_dir}{self.out_basename}_"
            f"kernel{int(self.kernel_width)}.{self.out_ext}"
        )
        hdf_out = h5py.File(self.outpath, "w")

        # Loop over all root keys apart from the mass grid itself
        for root_key in ["Parent", "Grid", "Units"]:
            grp = hdf_out.create_group(root_key)
            for key in hdf_rank0[root_key].attrs.keys():
                grp.attrs[key] = hdf_rank0[root_key].attrs[key]
        hdf_rank0.close()

        # Define the full grid shape
        grid_shape = tuple(self.cdim)

        # Create an empty resizable datasets to store the full grid and stds.
        # This lets us only load a slice at the time but write a single
        # grid
        dset = hdf_out.create_dataset(
            "KernelOverdensity",
            shape=grid_shape,
            maxshape=(None,) + grid_shape[1:],
            chunks=True,
            compression="gzip",
        )
        dset.attrs["Units"] = "dimensionless"
        dset.attrs[
            "Description"
        ] = "A 3D gird of overdensity expressed as log10(1 + delta)"
        std_dset = hdf_out.create_dataset(
            "KernelOverdensityStDev",
            shape=grid_shape,
            maxshape=(None,) + grid_shape[1:],
            chunks=True,
            compression="gzip",
        )
        std_dset.attrs["Units"] = "dimensionless"
        std_dset.attrs["Description"] = (
            "A 3D gird containing the standard deviation of the grid cell "
            "level overdensities within the kernel."
        )

        # Loop over the other ranks adding slices to the array
        slice_start = 0
        for other_rank in range(self.nranks):
            # Open this ranks file
            rankfile = (
                f"{self.out_dir}{self.out_basename}_kernel"
                f"{int(self.kernel_width)}_rank"
                f"{other_rank}.{self.out_ext}"
            )
            hdf_rank = h5py.File(rankfile, "r")

            # Get this ranks slice of the mass grid
            grid_slice = hdf_rank["KernelOverdensity"][...]
            std_grid_slice = hdf_rank["KernelOverdensityStDev"][...]

            # Calculate indices
            slice_end = slice_start + grid_slice.shape[0]

            # Add the low pad and slice itself
            dset[slice_start:slice_end, :, :] = grid_slice
            std_dset[slice_start:slice_end, :, :] = std_grid_slice

            hdf_rank.close()

            # Update the indices
            slice_start = slice_end

            # Delete the distributed file if we have been told to
            if delete_distributed:
                os.remove(rankfile)

        hdf_out.close()
