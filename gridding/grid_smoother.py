"""
Module for sorting particles into a 3D overdensity grid.

Example usage:

    gridder = GridGenerator("input.hdf5", "output.hdf5", 2.0)
    gridder.domain_decomp()
    gridder.get_grid()

This module provides a class, RegionGenerator, that generates all possible
regions smoothed over a spherical top hat kernel on a regular grid of
region centres. It includes methods for initializing the generator, setting
up MPI communication, reading simulation metadata, generating overdensity
grids from HDF5 simulation data, combining distributed files, and more.
"""

import os
from functools import lru_cache
import h5py
import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI


class GridGenerator:
    """
    Generates overdensity grid smoothed over a series of spherical top hat
    kernels.

    Attributes:
        input_path (str):
            Path to the input HDF5 file.
        outpath (str):
            Path to the output HDF5 file.
        out_basename (str):
            Base name for the output file, extracted from the outpath.
        out_ext (str):
            Extension of the output file, extracted from the outpath.
        out_dir (str):
            Directory of the output file, extracted from the outpath.
        comm (MPI.Comm):
            MPI communicator.
        nranks (int):
            Number of MPI ranks.
        rank (int):
            Rank of the current MPI process.
        nthreads (int):
            Number of threads.
        grid_width (float):
            Width of the region grid cells.
        boxsize (numpy.ndarray):
            Size of the simulation box in each dimension.
        redshift (float):
            Redshift of the simulation.
        nparts (int):
            Number of particles in the simulation.
        pmass (float):
            Particle mass.
        sim_cdim (int):
            Dimension of the simulation cells.
        ncells (int):
            Number of simulation cells.
        sim_width (numpy.ndarray):
            Size of the simulation cells in each dimension.
        half_sim_width (float):
            Half of the simulation cell width.
        mean_density (float):
            Mean density of the universe.
        grid_cdim (numpy.ndarray):
            Dimension of the region grid.
        grid_cell_volume (float):
            Volume of a grid cell.
        grid_ncells (int):
            Total number of cells in the region grid.
        kernel_width (float):
            Width of the spherical top hat kernel.
        kernel_rad (float):
            Radius of the spherical top hat kernel.
        kernel_vol (float):
            Volume of the spherical top hat kernel.
        my_grid_points (list):
            List of grid points assigned to the current MPI rank.
        x_ncells_rank (numpy.ndarray):
            Number of cells assigned to each MPI rank in the x-direction.
        tree (scipy.spatial.cKDTree):
            KDTree for particle data.
        masses (numpy.ndarray):
            Array of particle masses.
    """

    def __init__(
        self,
        inpath,
        outpath,
        region_grid_width,
        kernel_width,
        nthreads=1,
    ):
        """
        Initialize the region generator.

        Args:
            inpath (str):
                Path to the input HDF5 file.
            outpath (str):
                Path to the output HDF5 file.
            region_grid_width (float):
                Width of the region grid cells.
            kernel_width (float):
                Width of the spherical top hat kernel.
            nthreads (int, optional):
                Number of threads. Defaults to 1.
        """

        # Basic I/O information
        self.input_path = inpath
        self.outpath = outpath
        self.out_basename = outpath.split("/")[-1].split(".")[0]
        self.out_basename += f"_kernel{str(kernel_width).replace('.', 'p')}"
        self.out_ext = outpath.split("/")[-1].split(".")[-1]
        self.out_dir = "/".join(outpath.split("/")[:-1]) + "/"

        # Get the MPI information we need
        self.comm = None
        self.nranks = None
        self.rank = None
        self._setup_mpi()
        self.nthreads = nthreads

        # The target grid cell width (later refined to tesselated the parent
        # volume)
        self.grid_width = region_grid_width

        # Simulation metadata we'll need (populated in _read_attrs)
        self.boxsize = None
        self.redshift = None
        self.nparts = None
        self.pmass = None
        self.sim_cdim = None
        self.sim_ncells = None
        self.sim_width = None
        self.half_sim_width = None
        self.mean_density = None

        # Grid meta data (populated in _read_attrs)
        self.grid_cdim = None
        self.grid_cell_volume = None
        self.grid_ncells = None
        self.grid_cdim_per_cell = None

        # Define kernel properties
        self.kernel_width = kernel_width
        self.kernel_rad = kernel_width / 2
        self.kernel_vol = 4 / 3 * np.pi * self.kernel_rad**3

        # Read the simulation attributes and calculate grid properties
        self._read_attrs()

        # Information about the domain decomposition
        self.rank_cells = None
        self.rank_ncells = None
        self.rank_grid_ncells = None
        self.rank_cells_low = None
        self.rank_cells_high = None

        # assert (
        #     self.grid_width[0] < self.kernel_rad
        # ), "grid spacing must be less than the kernel radius"

    def _setup_mpi(self):
        """
        Sets up MPI communication.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

    def _read_attrs(self):
        """
        Read simulation metadata and calculate grid properties.
        """

        with h5py.File(self.input_path, "r") as hdf:
            # Read simulation metadata
            self.boxsize = hdf["Header"].attrs["BoxSize"]
            self.redshift = hdf["Header"].attrs["Redshift"]
            self.nparts = hdf["/PartType1/Masses"].size
            self.pmass = hdf["Header"].attrs["InitialMassTable"][1]
            self.sim_cdim = hdf["Cells/Meta-data"].attrs["dimension"]
            self.sim_ncells = np.prod(self.sim_cdim)
            self.sim_width = hdf["Cells/Meta-data"].attrs["size"]
            self.half_sim_width = self.sim_width / 2

        # Calculate the (comoving) mean density of the universe in
        # 10 ** 10 Msun / Mpc ** 3
        tot_mass = self.nparts * self.pmass
        self.mean_density = tot_mass / (
            self.boxsize[0] * self.boxsize[1] * self.boxsize[2]
        )

        # Compute the grid cell properties
        self.grid_cdim_per_cell = (
            np.int64(self.sim_width / self.grid_width) + 1
        )
        self.grid_cdim = self.grid_cdim_per_cell * self.sim_cdim
        self.grid_width = self.boxsize / self.grid_cdim
        self.grid_cell_volume = (
            self.grid_width[0] * self.grid_width[1] * self.grid_width[2]
        )
        self.grid_ncells = (
            self.grid_cdim[0] * self.grid_cdim[1] * self.grid_cdim[2]
        )

        if self.grid_cdim[0] == 0:
            raise ValueError(
                "Found 0 grid cells, decrease the grid cell width"
            )

        if self.rank == 0:
            print("PARENT METADATA:")
            print("Boxsize:", self.boxsize)
            print("CDim:", self.sim_cdim)
            print("Redshift:", self.redshift)
            print("Npart:", self.nparts)
            print("Particle Mass:", self.pmass)
            print("Number of simulation cells:", self.sim_ncells)
            print("Mean Density:", self.mean_density)
            print("Sim Cell Width:", self.sim_width)
            print()
            print("GRID METADATA:")
            print("Grid Cell Width:", self.grid_width)
            print(
                "Grid Cell Volume:",
                self.grid_width[0] * self.grid_width[1] * self.grid_width[2],
            )
            print("Grid CDim:", self.grid_cdim)
            print("N Grid cells:", self.grid_ncells)
            print("N Grid cells in SWIFT cell", self.grid_cdim_per_cell)
            print()

    def get_grid_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a grid cell.

        Args:
            cid (int):
                The flattened index.

        Returns:
            Tuple[int, int, int]:
                The i, j, k coordinates.
        """
        i = int(cid / (self.grid_cdim[1] * self.grid_cdim[2]))
        j = int((cid / self.grid_cdim[2]) % self.grid_cdim[1])
        k = int(cid % self.grid_cdim[2])

        return i, j, k

    def get_grid_cellid(self, i, j, k):
        """
        Compute the flattened index of a grid cell coordinates.

        Args:
            i (int):
                The integer i coordinate.
            j (int):
                The integer j coordinate.
            k (int):
                The integer k coordinate.

        Returns:
            int:
                The flattened index.
        """
        return k + self.grid_cdim[2] * (j + self.grid_cdim[1] * i)

    def get_sim_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a SWIFT cell.

        Args:
            cid (int):
                The flattened index.

        Returns:
            Tuple[int, int, int]:
                The i, j, k coordinates.
        """
        i = int(cid / (self.sim_cdim[1] * self.sim_cdim[2]))
        j = int((cid / self.sim_cdim[2]) % self.sim_cdim[1])
        k = int(cid % self.sim_cdim[2])

        return i, j, k

    def get_sim_cellid(self, i, j, k):
        """
        Compute the flattened index of a SWIFT cell coordinates.

        Args:
            i (int):
                The integer i coordinate.
            j (int):
                The integer j coordinate.
            k (int):
                The integer k coordinate.

        Returns:
            int:
                The flattened index.
        """
        return k + self.sim_cdim[2] * (j + self.sim_cdim[1] * i)

    def domain_decomp(self):
        """
        Divide cells into Nranks slices along the i direction. We don't care
        about the exact weight of each slice.
        """

        # Split theSWIFT cells amongst all ranks
        self.rank_cells = np.linspace(
            0,
            self.sim_ncells,
            self.nranks + 1,
            dtype=int,
        )

        # How many SWIFT cells are on each rank?
        self.rank_ncells = self.rank_cells[1:] - self.rank_cells[:-1]

        # Get this ranks cells
        self.my_cells_low = self.rank_cells[self.rank]
        self.my_cells_high = self.rank_cells[self.rank + 1]

    def _create_rank_output(self):
        """
        Create an output file for the current rank.
        """

        # Create the output file
        with h5py.File(
            f"{self.out_dir}{self.out_basename}_rank{self.rank}"
            f".{self.out_ext}",
            "w",
        ) as hdf_out:
            # Write out attributes about the parent simulation
            parent = hdf_out.create_group("Parent")
            parent.attrs["Boxsize"] = self.boxsize
            parent.attrs["Redshift"] = self.redshift
            parent.attrs["Npart"] = self.nparts
            parent.attrs["CDim"] = self.sim_cdim
            parent.attrs["CellWidth"] = self.sim_width
            parent.attrs["DMMass"] = self.pmass
            parent.attrs["MeanDensity"] = self.mean_density

            # Write out grid attributes
            ovden_grid_grp = hdf_out.create_group("Grid")
            ovden_grid_grp.attrs["CellWidth"] = self.grid_width
            ovden_grid_grp.attrs["CellVolume"] = self.grid_cell_volume
            ovden_grid_grp.attrs["CDim"] = self.grid_cdim
            ovden_grid_grp.attrs["KernelWidth"] = self.kernel_width
            ovden_grid_grp.attrs["KernelRadius"] = self.kernel_rad
            ovden_grid_grp.attrs["KernelVolume"] = self.kernel_vol

            # Make the group to hold cell grid points
            hdf_out.create_group("CellGrids")

            # Open the input file
            with h5py.File(self.input_path, "r") as hdf:
                # Store some unit information
                units = hdf_out.create_group("Units")
                for key in hdf["Units"].attrs.keys():
                    units.attrs[key] = hdf["Units"].attrs[key]

    def _read_particle_data(self, input_path, cids):
        coords = []
        masses = []
        with h5py.File(input_path, "r") as hdf:

            @lru_cache(maxsize=1000)
            def _single_cell_read(cid):
                """
                Get the particles from a single cell.

                This uses a cache to remember cells recently read and avoid
                unnecessary IO.
                """
                # Get the cell look up table data
                offset = hdf["/Cells/OffsetsInFile/PartType1"][cid]
                count = hdf["/Cells/Counts/PartType1"][cid]

                if count == 0:
                    return [], []

                # Store the indices
                part_indices = list(range(offset, offset + count))

                return (
                    hdf["/PartType1/Coordinates"][part_indices, :],
                    hdf["/PartType1/Masses"][part_indices],
                )

            # Loop over the cells we need particles from
            for cid in cids:
                # Read the particle data
                cell_coords, cell_masses = _single_cell_read(cid)
                coords.extend(cell_coords)
                masses.extend(cell_masses)

        return np.array(coords), np.array(masses)

    def get_grid(self):
        """
        Generates overdensity grids from HDF5 simulation data.
        """

        # Create the output file and populate the attributes we need
        self._create_rank_output()

        # Make a set of grid indices for a single simulation cell,
        ii, jj, kk = np.meshgrid(
            np.linspace(
                0,
                self.grid_cdim_per_cell[0] - 1,
                self.grid_cdim_per_cell[0],
                dtype=int,
            ),
            np.linspace(
                0,
                self.grid_cdim_per_cell[1] - 1,
                self.grid_cdim_per_cell[1],
                dtype=int,
            ),
            np.linspace(
                0,
                self.grid_cdim_per_cell[2] - 1,
                self.grid_cdim_per_cell[2],
                dtype=int,
            ),
        )
        grid_indices = np.zeros((ii.size, 3), dtype=int)
        grid_indices[:, 0] = ii.flatten()
        grid_indices[:, 1] = jj.flatten()
        grid_indices[:, 2] = kk.flatten()

        # Convert the indices into coordinates, we'll shift this
        # for each cell below
        grid_coords = (
            np.array(
                grid_indices,
                dtype=np.float32,
            )
            + 0.5
        ) * self.grid_width

        # Define how many cells we need to walk out
        delta_ijk = int(np.floor(self.kernel_rad / self.sim_width[0])) + 1

        # Loop over SWIFT cells, for each cell we'll get the neighbouring
        # cells, get their particles, construct a tree, populate the grid,
        # and finally write that subgrid to a HDF5 group
        for cid in range(self.my_cells_low, self.my_cells_high):
            print(
                self.rank,
                f"{(cid - self.my_cells_low)/(self.my_cells_high - self.my_cells_low) * 100}%",
            )
            # Get the simulation integer coordinates of this cell
            i, j, k = self.get_sim_cell_ijk(cid)

            # Get a list of the cells we'll need
            sim_cells = []

            # Get neighbouring cells out to distance covered by the kernel
            for ii in range(i - delta_ijk, i + delta_ijk + 1):
                ii = (ii + self.sim_cdim[0]) % self.sim_cdim[0]
                for jj in range(j - delta_ijk, j + delta_ijk + 1):
                    jj = (jj + self.sim_cdim[1]) % self.sim_cdim[1]
                    for kk in range(k - delta_ijk, k + delta_ijk + 1):
                        kk = (kk + self.sim_cdim[2]) % self.sim_cdim[2]
                        sim_cells.append(self.get_sim_cellid(ii, jj, kk))

            # Get the particle coordinates and masses for these cells, as
            # well as the edge of cid
            coords, masses = self._read_particle_data(
                self.input_path, sim_cells
            )

            # Construct the tree for these cells
            tree = cKDTree(coords, boxsize=self.boxsize)

            # Get the actucal grid point positions
            this_grid_coords = grid_coords + np.array(
                [
                    i * self.sim_width[0],
                    j * self.sim_width[1],
                    k * self.sim_width[2],
                ]
            )

            # Query the tree and get all particles associated to each
            # grid point
            part_queries = tree.query_ball_point(
                this_grid_coords,
                r=self.kernel_rad,
                workers=self.nthreads,
            )

            # Calculate 1 + delta for each grid point and store it
            grid_masses = np.array(
                list(map(lambda inds: np.sum(masses[inds]), part_queries))
            )
            grid_ovdens = (grid_masses / self.kernel_vol) / self.mean_density

            # Store the overdensities in a 3D array
            grid = np.zeros(tuple(self.grid_cdim_per_cell), dtype=np.float32)
            grid[
                grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]
            ] = grid_ovdens

            # Now store it in the rank file in a group for this cell
            with h5py.File(
                f"{self.out_dir}{self.out_basename}_rank{self.rank}"
                f".{self.out_ext}",
                "r+",
            ) as hdf_out:
                cell_grp = hdf_out["CellGrids"].create_group(f"{i}_{j}_{k}")

                # Write out this rank's grid points
                dset = cell_grp.create_dataset(
                    "OverDensity",
                    data=grid,
                    shape=grid.shape,
                    dtype=grid.dtype,
                    compression="gzip",
                )
                dset.attrs["Units"] = "dimensionless"
                dset.attrs["Description"] = (
                    "An array containing the overdensity in terms of 1 + delta"
                    "for a SWIFT simulation cell."
                )

    def combine_distributed_files(self, delete_distributed=False):
        """
        Create a single output file containing all slices.

        This method should only be called by rank 0 but protections are in
        place in case this isn't the case.

        Args:
            delete_distributed (bool, optional):
                Whether to remove the distributed
                files as they are processed. Defaults to False.
        """
        # Early exit if were not on rank 0
        if self.rank != 0:
            return

        # Open the rank 0 distributed file to get attributes
        rank0file = f"{self.out_dir}{self.out_basename}_rank0.{self.out_ext}"
        hdf_rank0 = h5py.File(rank0file, "r")

        # Create the single output file
        hdf_out = h5py.File(
            f"{self.out_dir}{self.out_basename}.{self.out_ext}", "w"
        )

        # Loop over all root keys apart from the overdensity grids themselves
        for root_key in {key for key in hdf_rank0.keys()} - set(["CellGrids"]):
            grp = hdf_out.create_group(root_key)
            for key in hdf_rank0[root_key].attrs.keys():
                grp.attrs[key] = hdf_rank0[root_key].attrs[key]
        hdf_rank0.close()

        # Create the group for each SWIFT cell's grid points
        cell_grids_grp = hdf_out.create_group("CellGrids")
        cell_grids_grp.attrs["Description"] = (
            "This group contains an overdensity grid split into each "
            "individual SWIFT cell. To extract a grid point's overdensity "
            "You need to first find the SWIFT cell the point is inside and "
            "then find the grid point relative to the the edge of the cell. "
            "A function for doing this is available in region_tools.py"
        )

        # Loop over all ranks adding their grid points to the file
        for other_rank in range(self.nranks):
            # Open this ranks file
            rankfile = (
                f"{self.out_dir}{self.out_basename}_rank{other_rank}"
                f".{self.out_ext}"
            )
            hdf_rank = h5py.File(rankfile, "r")

            # Get the cell grids group
            other_cell_grid_grp = hdf_rank["CellGrids"]

            # Loop over the cells adding their grid to the file
            for key in other_cell_grid_grp.keys():
                this_grid = other_cell_grid_grp[key]["OverDensity"][:]
                cell_grp = cell_grids_grp.create_group(key)
                dset = cell_grp.create_dataset(
                    "OverDensity",
                    data=this_grid,
                    shape=this_grid.shape,
                    dtype=this_grid.dtype,
                    compression="gzip",
                )
                dset.attrs["Units"] = "dimensionless"
                dset.attrs["Description"] = (
                    "An array containing the overdensity in terms of 1 + delta"
                    "for a SWIFT simulation cell."
                )

                # Include the edge of this cell for ease of use later
                i, j, k = [int(n) for n in key.split("_")]
                cell_grp.attrs["CellEdge"] = np.array(
                    [
                        i * self.sim_width[0],
                        j * self.sim_width[1],
                        k * self.sim_width[2],
                    ]
                )

            hdf_rank.close()

            # Delete the distributed file if we have been told to
            if delete_distributed:
                os.remove(rankfile)

        hdf_out.close()
