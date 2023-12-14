""" A module for sorting particle into a 3D mass histogram.

Example usage:

    gridder = GridGenerator("input.hdf5", "output.hdf5", 2.0)
    gridder.domain_decomp()
    gridder.get_grid()
"""
import os

import h5py
import numpy as np
from mpi4py import MPI


class GridGenerator:
    """
    Class to generate overdensity grids from SWIFT simulation data using MPI.

    Attributes:
        filepath (str):
            Path to the input HDF5 file.
        outpath (str):
            Path to the output HDF5 file.
        nranks (int):
            Number of MPI processes.
        rank (int):
            Rank of the current MPI process.
        target_grid_width (float):
            Target width for the overdensity grid cells.
    """

    def __init__(
        self,
        inpath,
        outpath,
        target_grid_width,
        pad_region=5,
    ):
        """
        Initializes the OverdensityGridGenerator.

        Args:
            filepath (str):
                Path to the input HDF5 file.
            outpath (str):
                Path to the output HDF5 file.
            nranks (int):
                Number of MPI processes.
            rank (int):
                Rank of the current MPI process.
            target_grid_width (float, optional):
                Target width for the overdensity grid cells. Default is 2.0.
            pad_region (int, optional):
                Number of cells to pad the region around each cell. Default is 2.
        """

        # Basic I/O information
        self.filepath = inpath
        self.outpath = outpath
        self.out_basename = outpath.split("/")[-1].split(".")[0]
        self.out_ext = outpath.split("/")[-1].split(".")[-1]
        self.out_dir = "/".join(outpath.split("/")[:-1]) + "/"

        # Get the MPI information we need
        self.comm = None
        self.nranks = None
        self.rank = None
        self._setup_mpi()

        # How much are we padding to capture particles outside their cells?
        self.pad_region = pad_region
        self.pad_ncells = 2 * pad_region

        # The target grid cell width (later refined to tesselated the parent
        # volume)
        self.target_grid_width = target_grid_width

        # Simulation metadata we'll need (populated in _read_attrs)
        self.boxsize = None
        self.redshift = None
        self.nparts = None
        self.pmass = None
        self.sim_cdim = None
        self.ncells = None
        self.cell_width = None
        self.half_cell_width = None
        self.mean_density = None

        # Grid meta data (populated in _read_attrs)
        self.grid_cdim = None
        self.grid_cell_width = None
        self.grid_per_sim_cells = None
        self.grid_cell_volume = None
        self.grid_per_sim_cells = None
        self.grid_ncells = None

        # Read the simulation attributes and calculate grid properties
        self._read_attrs()

        # Information about the domain decomposition
        self.my_cells = []
        self.x_cells_rank = None

    def _setup_mpi(self):
        """
        Sets up MPI communication.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

    def _read_attrs(self):
        """ """

        hdf = h5py.File(self.filepath, "r")

        # Read simulation metadata
        self.boxsize = hdf["Header"].attrs["BoxSize"]
        self.redshift = hdf["Header"].attrs["Redshift"]
        self.nparts = hdf["/PartType1/Masses"].size
        self.pmass = hdf["Header"].attrs["InitialMassTable"][1]
        self.sim_cdim = hdf["Cells/Meta-data"].attrs["dimension"]
        self.ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
        self.cell_width = hdf["Cells/Meta-data"].attrs["size"]
        self.half_cell_width = self.cell_width / 2
        hdf.close()

        # Calculate the (comoving) mean density of the universe in
        # 10 ** 10 Msun / Mpc ** 3
        tot_mass = self.nparts * self.pmass
        self.mean_density = tot_mass / (
            self.boxsize[0] * self.boxsize[1] * self.boxsize[2]
        )

        # Compute the grid cell properties
        self.grid_per_sim_cells = np.int32(self.cell_width / self.target_grid_width)
        self.grid_cell_width = self.cell_width / self.grid_per_sim_cells
        self.grid_cdim = self.grid_per_sim_cells * self.sim_cdim
        self.grid_cell_volume = (
            self.grid_cell_width[0] * self.grid_cell_width[1] * self.grid_cell_width[2]
        )
        self.grid_ncells = self.grid_cdim * self.grid_cdim * self.grid_cdim

        if self.grid_ncells[0] == 0:
            raise ValueError("Found 0 grid cells, decrease your cell_width")

        if self.rank == 0:
            print("PARENT METADATA:")
            print("Boxsize:", self.boxsize)
            print("CDim:", self.sim_cdim)
            print("Redshift:", self.redshift)
            print("Npart:", self.nparts)
            print("Particle Mass:", self.pmass)
            print("Number of simulation cells:", self.ncells)
            print("Mean Density:", self.mean_density)
            print("Sim Cell Width:", self.cell_width)
            print()
            print("GRID METADATA:")
            print("Grid Cell Width:", self.grid_cell_width)
            print(
                "Grid Cell Volume:",
                self.grid_cell_width[0]
                * self.grid_cell_width[1]
                * self.grid_cell_width[2],
            )
            print("N grid cell per simulation cell", self.grid_per_sim_cells)
            print("Parent Grid NCells:", self.grid_cdim)
            print()

    def get_sim_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a grid cell.

        Args:
            cid (int)
                The flattened index.
        """
        i = int(cid / (self.sim_cdim[1] * self.sim_cdim[2]))
        j = int((cid / self.sim_cdim[2]) % self.sim_cdim[1])
        k = int(cid % self.sim_cdim[2])

        return i, j, k

    def get_sim_cellid(self, i, j, k):
        """
        Compute the flattened index of a grid/simulation cell coordinates.

        Args:
            i, j, k (int)
                The integer cell coordinates.
        """
        return k + self.sim_cdim[2] * (j + self.sim_cdim[1] * i)

    def domain_decomp(self):
        """
        Divide cells into Nranks slices along the i direction. We don't care
        about the exact weight of each slice.
        """

        assert (
            self.nranks <= self.sim_cdim[0]
        ), "Can't have more ranks than SWIFT cells."

        # Split the x direction amongst all ranks
        rank_cells = np.linspace(
            0,
            self.sim_cdim[0] - 1,
            self.nranks + 1,
            dtype=int,
        )

        # How many x cells are on this rank?
        self.x_cells_rank = rank_cells[self.rank + 1] - rank_cells[self.rank]

        # Loop over all cells and construct lists to be sliced
        for i in range(rank_cells[self.rank], rank_cells[self.rank + 1]):
            for j in range(self.sim_cdim[1]):
                for k in range(self.sim_cdim[2]):
                    cell = self.get_sim_cellid(i, j, k)
                    self.my_cells.append(cell)

        # Convert the list to an array
        self.my_cells = np.array(self.my_cells)

        print("Rank=", self.rank, "- My Ncells=", len(self.my_cells))

    def _create_output(self):
        """ """

        # Create the output file
        hdf_out = h5py.File(
            f"{self.out_dir}{self.out_basename}_rank{self.rank}.{self.out_ext}", "w"
        )

        # Write out attributes about the parent simulation
        parent = hdf_out.create_group("Parent")
        parent.attrs["Boxsize"] = self.boxsize
        parent.attrs["Redshift"] = self.redshift
        parent.attrs["Npart"] = self.nparts
        parent.attrs["CDim"] = self.sim_cdim
        parent.attrs["CellWidth"] = self.cell_width
        parent.attrs["DMMass"] = self.pmass
        parent.attrs["MeanDensity"] = self.mean_density

        # Write out grid attributes
        ovden_grid_grp = hdf_out.create_group("Grid")
        ovden_grid_grp.attrs["CellWidth"] = self.grid_cell_width
        ovden_grid_grp.attrs["CDim"] = self.grid_cdim
        ovden_grid_grp.attrs["NcellsPerSimCell"] = self.grid_per_sim_cells

        # Open the input file
        hdf = h5py.File(self.filepath, "r")

        # Store some unit information
        units = hdf_out.create_group("Units")
        for key in hdf["Units"].attrs.keys():
            units.attrs[key] = hdf["Units"].attrs[key]

        hdf.close()
        hdf_out.close()

    def get_grid(self):
        """
        Generates overdensity grids from HDF5 simulation data.
        """

        # Create the output file and populate the attributes we need
        self._create_output()

        # Set up the grid for this rank's slice
        mass_grid = np.zeros(
            (
                (self.x_cells_rank * self.grid_per_sim_cells[0]) + self.pad_ncells,
                self.grid_cdim[1],
                self.grid_cdim[2],
            ),
            dtype=np.float32,
        )

        # Open the input file
        hdf = h5py.File(self.filepath, "r")

        # Lets get the indices of all the particles we will need and do one big
        # read at once
        cells = {
            "indices": [],
            "pointers": np.zeros(len(self.my_cells), dtype=int),
            "counts": np.zeros(len(self.my_cells), dtype=int),
            "edges": np.zeros((len(self.my_cells), 3)),
        }
        for ind, my_cell in enumerate(self.my_cells):
            # Get the cell look up table data
            my_offset = hdf["/Cells/OffsetsInFile/PartType1"][my_cell]
            my_count = hdf["/Cells/Counts/PartType1"][my_cell]
            my_edges = hdf["/Cells/Centres"][my_cell, :] - self.half_cell_width

            # Store the pointer
            cells["pointers"][ind] = len(cells["indices"])

            # Store the count
            cells["counts"][ind] = my_count

            # Store the edges
            cells["edges"][ind, :] = my_edges

            # Store the indices
            if my_count > 0:
                cells["indices"].extend(list(range(my_offset, my_offset + my_count)))

        # Read the particle data
        all_poss = hdf["/PartType1/Coordinates"][cells["indices"], :]
        all_masses = hdf["/PartType1/Masses"][cells["indices"]]

        # No longer need the indices
        del cells["indices"]

        # Loop over the cells on this rank and grid the particles
        for ind, my_cell in enumerate(self.my_cells):
            # Get the cell look up table data
            my_edges = cells["edges"][ind]
            my_count = cells["counts"][ind]
            start = cells["pointers"][ind]
            end = start + my_count

            # No point look for particles if the cell is empty
            if my_count > 0:
                # Get particle positions
                poss = all_poss[start:end, :]

                # Shift the positions to account for the slice edge
                poss[:, 0] -= my_edges[0] - (self.pad_region * self.grid_cell_width[0])

                # Get particle masses
                masses = all_masses[start:end]

                # Convert positions into grid cell indices
                ijk = np.int64(poss / self.grid_cell_width)

                # Wrap the x and y indices, the x axis is padded
                ijk[:, 1] = (ijk[:, 1] + self.grid_cdim[1]) % self.grid_cdim[1]
                ijk[:, 2] = (ijk[:, 2] + self.grid_cdim[2]) % self.grid_cdim[2]

                # Add the particle mass to the grid cells
                mass_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] += masses

                print(
                    np.any(
                        np.int64(
                            poss
                            + (self.pad_region * self.grid_cell_width[0])
                            / self.grid_cell_width
                        )
                        < 0
                    ),
                    np.any(
                        np.int64(
                            poss
                            + (self.pad_region * self.grid_cell_width[0])
                            / self.grid_cell_width
                        )
                        > mass_grid.shape[0]
                    ),
                )

        hdf.close()

        # Open the output file
        hdf_out = h5py.File(
            f"{self.out_dir}{self.out_basename}_rank{self.rank}.{self.out_ext}",
            "r+",
        )

        # Store the x edge of this rank (we'll need this to combine the grids
        # later on)
        hdf_out.attrs["SliceEdge"] = my_edges[0]

        # Write out this slice of the mass file
        dset = hdf_out.create_dataset(
            "MassGrid",
            data=mass_grid,
            shape=mass_grid.shape,
            dtype=mass_grid.dtype,
            compression="gzip",
        )
        dset.attrs["Units"] = "1e10 Msun"
        dset.attrs["Description"] = (
            "A 3D mass histogram of a slice of the parent "
            "simulation dark matter distribution in 10 ** 10 Msun."
        )

        hdf_out.close()

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
        rank0file = f"{self.out_dir}{self.out_basename}_rank0.{self.out_ext}"
        hdf_rank0 = h5py.File(rank0file, "r")

        # Create the single output file
        hdf_out = h5py.File(self.outpath, "w")

        # Loop over all root keys apart from the mass grid itself
        for root_key in {key for key in hdf_rank0.keys()} - set(["MassGrid"]):
            grp = hdf_out.create_group(root_key)
            for key in hdf_rank0[root_key].attrs.keys():
                grp.attrs[key] = hdf_rank0[root_key].attrs[key]
        hdf_rank0.close()

        # Define the full grid shape
        grid_shape = tuple(self.grid_cdim)

        # Create an empty resizable dataset to store the full mass grid.
        # This lets us only load a slice at the time but write a single
        # grid
        dset = hdf_out.create_dataset(
            "MassGrid",
            shape=grid_shape,
            maxshape=(None,) + grid_shape[1:],
            chunks=(self.x_cells_rank, self.grid_cdim[1], self.grid_cdim[2]),
            compression="gzip",
        )
        dset.attrs["Units"] = "1e10 Msun"
        dset.attrs["Description"] = (
            "A 3D mass histogram of a slice of the parent "
            "simulation dark matter distribution in 10 ** 10 Msun."
        )

        # Loop over the other ranks adding slices to the array
        low_start = -self.pad_region
        low_end = grid_shape[0] if self.pad_region != 0 else 0
        slice_start = 0
        high_start = 0
        high_end = 0
        for other_rank in range(self.nranks):
            # Open this ranks file
            rankfile = (
                f"{self.out_dir}{self.out_basename}_rank" f"{other_rank}.{self.out_ext}"
            )
            hdf_rank = h5py.File(rankfile, "r")

            # Get this ranks slice of the mass grid
            grid_slice = hdf_rank["MassGrid"][...]

            # Get the padded areas of the slice
            pad_low = grid_slice[: self.pad_region :, :, :]
            slice_mid = grid_slice[self.pad_region : -self.pad_region, :, :]
            pad_up = grid_slice[-self.pad_region :, :, :]

            # Calculate indices
            slice_end = slice_start + slice_mid.shape[0]
            high_start += slice_mid.shape[0]
            high_end = high_start + pad_up.shape[0]
            high_start %= grid_shape[0]
            high_end %= grid_shape[0]

            print(
                f"{low_start}-{low_end}, {slice_start}-{slice_end}, {high_start}-{high_end}"
            )

            # Add the low pad and slice itself
            dset[low_start:low_end, :, :] += pad_low
            dset[slice_start:slice_end, :, :] += slice_mid
            dset[high_start:high_end, :, :] += pad_up

            hdf_rank.close()

            # Update the indices
            slice_start = slice_end
            low_start = slice_start - self.pad_region
            low_end = slice_start

            # Delete the distributed file if we have been told to
            if delete_distributed:
                os.remove(rankfile)

        hdf_out.close()
