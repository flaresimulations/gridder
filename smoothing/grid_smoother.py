""" A module for smoothing grids over a spherical kernel.

This expects a file in the the output format produced by ParentGridder. It will
smooth every grid cell over an arbitrarily size spherical top hat kernel and
produce a distributed set of files followed by a single combined file.

Examples usage:

    # Create the grid instance
    smoother = GridSmoother(float(sys.argv[4]), out_dir + metafile)

    # Decompose the grid
    smoother.decomp_cells(rank, nranks)

    # Compute the smoothed grid
    smoother.smooth_grid_cells(rank)

    # Output the grid from each rank to the distributed files
    smoother.write_smoothed_grid_rankfile(outpath, rank)

    comm.Barrier()

    # Convert the distributed files into a single file
    smoother.write_smoothed_grid_meta(outpath, metafile, rank, nranks)

"""
import h5py
import numpy as np


class GridSmoother:
    """
    A class for smoothing grids over a spherical kernel.

    Args:
        kernel_width (float):
            Width of the spherical top hat kernel.
        grid_file (str):
            Path to the grid file in the format produced by ParentGridder.
        rank (int):
            The rank ID of the current rank.

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
        region_vals (numpy.ndarray):
            Array to store smoothed region values.
        region_stds (numpy.ndarray):
            Array to store smoothed region standard deviations.
        region_inds (numpy.ndarray):
            Array to store indices of smoothed regions.
        centres (numpy.ndarray):
            Array to store the center coordinates of smoothed regions.

    """

    def __init__(self, kernel_width, grid_file, rank):
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

        # Set up the grid
        self.grid_file = grid_file

        # Set up variables to be populated when opening the grid
        self.cell_width = None
        self.boxsize = None
        self.mean_density = None
        self.ngrid_cells = None
        self.cdim = None

        # Read the grid attributes
        self._read_attrs(rank)

        # The mass grid dataset
        self.mass_gird = None

        # Define kernel properties
        self.kernel_width = kernel_width
        self.kernel_rad = kernel_width / 2

        # How many cells are in a kernel?
        self.cells_in_kernel = int(np.ceil(self.kernel_width / self.cell_width)) + 1

        # Define the decomposition attributes
        self.rank_cells = None
        self.rank_ncells = None

        # Set up the outputs to write out
        self.region_vals = None
        self.region_stds = None
        self.region_inds = None
        self.centres = None

    def _read_attrs(self, rank):
        """
        Read the grid file for attributes.


        """

        # Open file
        hdf = h5py.File(self.grid_file, "r")

        # Get metadata
        self.boxsize = hdf["Parent"].attrs["Boxsize"]
        self.mean_density = hdf["Parent"].attrs["Mean_Density"]
        self.ngrid_cells = hdf["Delta_grid"].attrs["Ncells_Total"]
        self.cell_width = hdf["Delta_grid"].attrs["Cell_Width"]

        # Print some nice things
        if rank == 0:
            print("Boxsize:", self.boxsize)
            print("Mean Density:", self.mean_density)
            print("Grid Cell Width:", self.cell_width)
            print("Grid cells total:", self.ngrid_cells)

        # Get results array dimensions
        self.cdim = hdf["Parent_Grid"].shape

        hdf.close()

    def _apply_spherical_top_hat(self, subset, func, centre):
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

        # Get the shape of the input array
        shape = np.array(subset.shape)

        # Create grid cell coordinates
        cell_coords = shape * self.cell_width
        xx, yy, zz = np.meshgrid(
            cell_coords - centre[0],
            cell_coords - centre[1],
            cell_coords - centre[2],
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

    def decomp_cells(self, rank, nranks):
        """
        Decompose the grid cells over N ranks.

        Args:
            rank (int)
                The rank id of this rank.
            nranks (int)
                The number of ranks.
        """
        # Find the cells and simulation ijk grid references
        # that this rank has to work on
        self.rank_cells = np.linspace(
            0,
            self.ngrid_cells,
            nranks + 1,
            dtype=int,
        )
        self.rank_ncells = self.rank_cells[rank + 1] - self.rank_cells[rank]

        print(
            f"Rank {rank} has {self.rank_ncells} cells",
        )

        # Find the minimum and maximum i, j and k to slice main grid
        min_i, min_j, min_k = np.inf, np.inf, np.inf
        max_i, max_j, max_k = 0, 0, 0
        for cid in range(self.rank_cells[rank], self.rank_cells[rank + 1]):
            i, j, k = self.get_cell_ijk(cid)
            if i < min_i:
                min_i = i
            if i > max_i:
                max_i = i
            if j < min_j:
                min_j = j
            if j > max_j:
                max_j = j
            if k < min_k:
                min_k = k
            if k > max_k:
                max_k = k

        print(
            f"Rank = {rank} has cells in range "
            "[{min_i}-{max_i}, {min_j}-{max_j}, {min_k}-{max_k}]"
        )

        # Store the grid cell limits
        self.min_i = min_i
        self.min_j = min_j
        self.min_k = min_k
        self.max_i = max_i
        self.max_j = max_j
        self.max_k = max_k

        # Define an array to store the indices we need to extract
        indices = np.zeros(
            (
                (max_i - min_i) + 2 * self.cells_in_kernel,
                (max_j - min_j) + 2 * self.cells_in_kernel,
                (max_k - min_k) + 2 * self.cells_in_kernel,
                3,
            ),
            dtype=int,
        )

        # Get the grid indices with buffer
        buffer_min_i = min_i - self.cells_in_kernel
        buffer_min_j = min_j - self.cells_in_kernel
        buffer_min_k = min_k - self.cells_in_kernel

        # Loop over the cells and get what grid indices we need with wrapping
        # and the buffer for the kernel
        for i in range((max_i - min_i) + 2 * self.cells_in_kernel):
            ii = (buffer_min_i + i + self.cdim) % self.cdim
            for j in range((max_i - min_i) + 2 * self.cells_in_kernel):
                jj = (buffer_min_j + j + self.cdim) % self.cdim
                for k in range((max_i - min_i) + 2 * self.cells_in_kernel):
                    kk = (buffer_min_k + i + self.cdim) % self.cdim

                    # We've wrapped everything so store the index
                    indices[i, j, k, 0] = ii
                    indices[i, j, k, 1] = jj
                    indices[i, j, k, 2] = kk

        # Open file to get the relevant grid cells
        hdf = h5py.File(path, "r")

        # Get the mass grid
        self.mass_grid = hdf["Parent_Grid"][
            indices[:, :, :, 0], indices[:, :, :, 1], indices[:, :, :, 2]
        ]

        hdf.close()

    def smooth_grid_cells(self, rank):
        """ """

        # Set up the outputs to write out
        self.region_vals = np.zeros(self.rank_ncells)
        self.region_stds = np.zeros(self.region_vals.size)
        self.region_inds = np.zeros(self.region_vals.size, dtype=int)
        self.centres = np.zeros((self.region_vals.size, 3))

        # Loop over this ranks cells
        ind = 0
        for cid in range(self.rank_cells[rank], self.rank_cells[rank + 1]):
            # Get the i j k coordinates
            i, j, k = self.get_cell_ijk(cid)

            # Calculate the centre
            centre = (np.array([i, j, k], dtype=np.float64) + 0.5) * self.cell_width

            # Shift is for this ranks subsample of the grid
            low_i = (
                i
                - self.min_i
                - self.cells_in_kernel
                - int(self.kernel_rad / self.cell_width)
                + 1
            )
            low_j = (
                j
                - self.min_j
                - self.cells_in_kernel
                - int(self.kernel_rad / self.cell_width)
                + 1
            )
            low_k = (
                k
                - self.min_k
                - self.cells_in_kernel
                - int(self.kernel_rad / self.cell_width)
                + 1
            )

            # Get the index for this smoothed grid cell in the full grid
            full_ind = self.get_cellid(i, j, k)

            # Get the mean of these overdensities
            sub_grid = self.mass_grid[
                low_i : low_i + self.cells_per_kernel,
                low_j : low_j + self.cells_per_kernel,
                low_k : low_k + self.cells_per_kernel,
            ]

            # Apply the kernel and get the standard deviation
            kernel_sum = self._apply_spherical_top_hat(sub_grid, np.sum, centre)
            kernel_std = self._apply_spherical_top_hat(sub_grid, np.std, centre)

            # Store edges
            self.centres[ind, :] = centre

            # Store the results and index
            self.region_vals[ind] = kernel_sum
            self.region_stds[ind] = kernel_std
            self.region_inds[ind] = full_ind

            ind += 1

    def write_smoothed_grid_rankfile(self, outpath, rank):
        """ """
        # Write out the results of smoothing
        outpath = outpath.split(".")[0] + f"_rank{rank}." + outpath.split(".")[-1]
        hdf = h5py.File(outpath, "w")
        hdf.attrs["Kernel_Width"] = self.kernel_width
        hdf.create_dataset(
            "Region_Overdensity",
            data=self.region_vals,
            shape=self.region_vals.shape,
            dtype=self.region_vals.dtype,
            compression="gzip",
        )
        hdf.create_dataset(
            "Region_Overdensity_Stdev",
            data=self.region_stds,
            shape=self.region_stds.shape,
            dtype=self.region_stds.dtype,
            compression="gzip",
        )
        hdf.create_dataset(
            "Region_Centres",
            data=self.centres,
            shape=self.centres.shape,
            dtype=self.centres.dtype,
            compression="gzip",
        )
        hdf.create_dataset(
            "Region_Indices",
            data=self.region_inds,
            shape=self.region_inds.shape,
            dtype=self.region_inds.dtype,
            compression="gzip",
        )

        hdf_grid = h5py.File(outpath, "r")

        for key in hdf_grid.attrs.keys():
            hdf.attrs[key] = hdf_grid.attrs[key]

        for root_key in ["Parent", "Delta_grid"]:
            for key in hdf_grid[root_key].attrs.keys():
                hdf.attrs[root_key + "_" + key] = hdf_grid[root_key].attrs[key]

        hdf_grid.close()

        hdf.close()

    def write_smoothed_grid_meta(self, outpath, metafile, rank, nranks):
        """ """

        if rank == 0:
            # Define arrays to store the collected results

            # Set up arrays for the smoothed grid
            final_region_vals = np.zeros(self.ngrid_cells)
            final_region_stds = np.zeros(final_region_vals.size)

            # Set up array to store centres
            final_centres = np.zeros((final_region_vals.size, 3))

            # Set up the outpaths
            outpath0 = outpath.split(".")[0] + "_rank0." + outpath.split(".")[-1]

            # Open file to combine results
            hdf = h5py.File(outpath, "w")

            # Open rank 0 file to get metadata
            hdf_rank0 = h5py.File(outpath0, "r")  # open rank 0 file
            for key in hdf_rank0.attrs.keys():
                hdf.attrs[key] = hdf_rank0.attrs[key]  # write attrs

            hdf_rank0.close()

            for other_rank in range(nranks):
                # Set up the outpath for each rank file
                rank_outpath = (
                    outpath.split(".")[0]
                    + f"_rank{other_rank}."
                    + outpath.split(".")[-1]
                )

                hdf_rank = h5py.File(rank_outpath, "r")  # open rank 0 file

                # Get indices
                inds = hdf_rank["Region_Indices"][...]

                # Combine this rank's results into the final array
                ods = hdf_rank["Region_Overdensity"][...]
                final_region_vals[inds] += ods

                final_region_stds[inds] += hdf_rank["Region_Overdensity_Stdev"][...]
                final_centres[inds] += hdf_rank["Region_Centres"][...]

                hdf_rank.close()

            sinds = np.argsort(final_region_vals)

            hdf.create_dataset(
                "Sorted_Indices",
                data=sinds,
                shape=sinds.shape,
                dtype=sinds.dtype,
                compression="gzip",
            )
            hdf.create_dataset(
                "Region_Overdensity",
                data=final_region_vals,
                shape=final_region_vals.shape,
                dtype=final_region_vals.dtype,
                compression="gzip",
            )
            hdf.create_dataset(
                "Region_Overdensity_Stdev",
                data=final_region_stds,
                shape=final_region_stds.shape,
                dtype=final_region_stds.dtype,
                compression="gzip",
            )
            hdf.create_dataset(
                "Region_Centres",
                data=final_centres,
                shape=final_centres.shape,
                dtype=final_centres.dtype,
                compression="gzip",
            )

            hdf.close()
