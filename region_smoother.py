""" A module for sorting particle into a 3D mass histogram.

Example usage:

    gridder = GridGenerator("input.hdf5", "output.hdf5", 2.0)
    gridder.domain_decomp()
    gridder.get_grid()
"""
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI


class RegionGenerator:
    """
    Generates all possible regions smoothed over a spherical top hat kernel
    on a regular grid of region centres.

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
        Intialise the region generator.
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
        self.ncells = None
        self.sim_width = None
        self.half_sim_width = None
        self.mean_density = None

        # Grid meta data (populated in _read_attrs)
        self.grid_cdim = None
        self.grid_cell_volume = None
        self.grid_ncells = None

        # Define kernel properties
        self.kernel_width = kernel_width
        self.kernel_rad = kernel_width / 2
        self.kernel_vol = 4 / 3 * np.pi * self.kernel_rad**3

        # Read the simulation attributes and calculate grid properties
        self._read_attrs()

        # Information about the domain decomposition
        self.rank_cells = None
        self.my_grid_points = None
        self.x_ncells_rank = None

        # Place holder attributes for the KDTree (to be populated during
        # domain decomposition)
        self.tree = None

        # We will need to store the particle masses
        self.part_masses = None

    def _setup_mpi(self):
        """
        Sets up MPI communication.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

    def _read_attrs(self):
        """ """

        hdf = h5py.File(self.input_path, "r")

        # Read simulation metadata
        self.boxsize = hdf["Header"].attrs["BoxSize"]
        self.redshift = hdf["Header"].attrs["Redshift"]
        self.nparts = hdf["/PartType1/Masses"].size
        self.pmass = hdf["Header"].attrs["InitialMassTable"][1]
        self.sim_cdim = hdf["Cells/Meta-data"].attrs["dimension"]
        self.ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
        self.sim_width = hdf["Cells/Meta-data"].attrs["size"]
        self.half_sim_width = self.sim_width / 2
        hdf.close()

        # Calculate the (comoving) mean density of the universe in
        # 10 ** 10 Msun / Mpc ** 3
        tot_mass = self.nparts * self.pmass
        self.mean_density = tot_mass / (
            self.boxsize[0] * self.boxsize[1] * self.boxsize[2]
        )

        # Compute the grid cell properties
        self.grid_cdim = np.int32(self.boxsize / self.grid_width)
        self.grid_width = self.boxsize / self.grid_cdim
        self.grid_cell_volume = (
            self.grid_width[0] * self.grid_width[1] * self.grid_width[2]
        )
        self.grid_ncells = self.grid_cdim[0] * self.grid_cdim[1] * self.grid_cdim[2]

        if self.grid_cdim[0] == 0:
            raise ValueError("Found 0 grid cells, decrease the grid cell width")

        if self.rank == 0:
            print("PARENT METADATA:")
            print("Boxsize:", self.boxsize)
            print("CDim:", self.sim_cdim)
            print("Redshift:", self.redshift)
            print("Npart:", self.nparts)
            print("Particle Mass:", self.pmass)
            print("Number of simulation cells:", self.ncells)
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
            print()

    def get_grid_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a grid cell.

        Args:
            cid (int)
                The flattened index.
        """
        i = int(cid / (self.grid_cdim[1] * self.grid_cdim[2]))
        j = int((cid / self.grid_cdim[2]) % self.grid_cdim[1])
        k = int(cid % self.grid_cdim[2])

        return i, j, k

    def get_grid_cellid(self, i, j, k):
        """
        Compute the flattened index of a grid cell coordinates.

        Args:
            i, j, k (int)
                The integer cell coordinates.
        """
        return k + self.grid_cdim[2] * (j + self.grid_cdim[1] * i)

    def get_sim_cell_ijk(self, cid):
        """
        Get the i, j, k coordinates of a SWIFT cell.

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
        Compute the flattened index of a SWIFT cell coordinates.

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

        # Split the x direction amongst all ranks
        rank_cells = np.linspace(
            0,
            self.grid_ncells,
            self.nranks + 1,
            dtype=int,
        )

        self.rank_cells = rank_cells

        # Create a range of grid points on this rank
        self.my_grid_points = range(
            rank_cells[self.rank],
            rank_cells[self.rank + 1],
        )

        # To get the simulation cell containing the kernel edges we need to
        # know how many grid points between the centre and edges
        delta_ijk = int(self.kernel_rad / self.grid_width[0]) + 1

        # Find the simulation grid cells covered by these grid points and
        # their associated
        sim_cells = set()
        sim_ijk = set()
        for gid in self.my_grid_points:
            # Get the simulation cell for the grid point itself
            i, j, k = self.get_grid_cell_ijk(gid)
            sim_i = int(i * self.grid_width[0] / self.sim_width[0])
            sim_j = int(j * self.grid_width[1] / self.sim_width[1])
            sim_k = int(k * self.grid_width[2] / self.sim_width[2])
            cid = self.get_sim_cellid(sim_i, sim_j, sim_k)
            sim_cells.update({cid})
            sim_ijk.update({(sim_i, sim_j, sim_k)})

            # Get the cell containing the kernel edges if different
            for i in [i - delta_ijk, i + delta_ijk]:
                ii = (i + self.grid_cdim[0]) % self.grid_cdim[0]
                sim_ii = int(ii * self.grid_width[0] / self.sim_width[0])
                if sim_i == sim_ii:
                    continue
                for j in [j - delta_ijk, j + delta_ijk]:
                    jj = (j + self.grid_cdim[1]) % self.grid_cdim[1]
                    sim_jj = int(jj * self.grid_width[1] / self.sim_width[1])
                    if sim_j == sim_jj:
                        continue
                    for k in [k - delta_ijk, k + delta_ijk]:
                        kk = (k + self.grid_cdim[2]) % self.grid_cdim[2]
                        sim_kk = int(kk * self.grid_width[2] / self.sim_width[2])
                        if sim_k == sim_kk:
                            continue
                        cid = self.get_sim_cellid(sim_ii, sim_jj, sim_kk)
                        sim_cells.update({cid})

        print(
            f"Rank {self.rank} - N_gridpoints = "
            f"{len(self.my_grid_points)} and N_simcells = {len(sim_cells)}"
        )

        # Get the coordinates and masses of particles in the cells covered
        # by the grid points

        # Open the input file
        with h5py.File(self.input_path, "r") as hdf:
            # Lets get the indices of all the particles we will need and
            # do one big read at once
            part_indices = []
            while len(sim_cells) > 0:
                cid = sim_cells.pop()

                # Get the cell look up table data
                offset = hdf["/Cells/OffsetsInFile/PartType1"][cid]
                count = hdf["/Cells/Counts/PartType1"][cid]

                if count == 0:
                    continue

                # Store the indices
                part_indices.extend(list(range(offset, offset + count)))

            # Sort the particle indices
            part_indices = np.sort(part_indices)

            # Read the particle data
            all_poss = hdf["/PartType1/Coordinates"][part_indices, :]
            self.part_masses = hdf["/PartType1/Masses"][part_indices]

        # Construct the KDTree
        self.tree = cKDTree(all_poss, leafsize=100, boxsize=self.boxsize)

    def _create_rank_output(self):
        """ """

        # Create the output file
        hdf_out = h5py.File(
            f"{self.out_dir}{self.out_basename}_rank{self.rank}.{self.out_ext}",
            "w",
        )

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

        # Open the input file
        hdf = h5py.File(self.input_path, "r")

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
        self._create_rank_output()

        # Set up the grid for this rank's slice
        grid = np.zeros(len(self.my_grid_points), dtype=np.float32)

        # Convert grid indices into grid coordinates
        grid_indices = np.array(
            [self.get_grid_cell_ijk(gid) for gid in self.my_grid_points],
            dtype=int,
        )
        grid_coords = (
            np.array(
                grid_indices,
                dtype=np.float32,
            )
            + 0.5
        ) * self.grid_width

        # Query the tree
        part_queries = self.tree.query_ball_point(
            grid_coords,
            r=self.kernel_rad,
            workers=self.nthreads,
        )

        # Calculate 1 + delta for each grid point and store it
        for ind, part_query in enumerate(part_queries):
            mass = np.sum(self.part_masses[part_query])
            grid[ind] = (mass / self.kernel_vol) / self.mean_density

        # Open the output file
        hdf_out = h5py.File(
            f"{self.out_dir}{self.out_basename}_rank{self.rank}.{self.out_ext}",
            "r+",
        )

        # Write out this rank's grid points
        dset = hdf_out.create_dataset(
            "OverDensity",
            data=grid,
            shape=grid.shape,
            dtype=grid.dtype,
            compression="gzip",
        )
        dset.attrs["Units"] = "dimensionless"
        dset.attrs["Description"] = (
            "A flattened array containing the overdensity in terms of 1 + delta"
            "for all the grid cells on this rank"
        )
        dset = hdf_out.create_dataset(
            "GridPoints",
            data=grid_indices,
            shape=grid_indices.shape,
            dtype=grid_indices.dtype,
            compression="gzip",
        )
        dset.attrs["Units"] = "dimensionless"
        dset.attrs["Description"] = "The integer cordinates of each grid point"

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
        for root_key in {key for key in hdf_rank0.keys()} - set(["OverDensity"]):
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
            "OverDensity",
            shape=(self.grid_ncells),
            chunks=True,
            compression="gzip",
        )
        dset.attrs["Units"] = "dimensionless"
        dset.attrs["Description"] = "A 3D grid of overdensities in terms of 1 + delta"

        # Loop over the other ranks adding their grid cells to the array
        for other_rank in range(self.nranks):
            # Open this ranks file
            rankfile = (
                f"{self.out_dir}{self.out_basename}_rank{other_rank}.{self.out_ext}"
            )
            hdf_rank = h5py.File(rankfile, "r")

            # Get the rank's overdensities
            grid = hdf_rank["OverDensity"][...]

            # Set this rank's grid points
            dset[self.rank_cells[other_rank] : self.rank_cells[other_rank + 1]] = grid

            hdf_rank.close()

            # Delete the distributed file if we have been told to
            if delete_distributed:
                os.remove(rankfile)

        # Finally reshape the grid to actually be 3D
        dset.resize(grid_shape)

        hdf_out.close()
