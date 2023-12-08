import os
import sys

import h5py
import numpy as np
from mpi4py import MPI

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def get_cellid(cdim, i, j, k):
    """
    Compute the flattened index of a grid/simulation cell coordinates. 
    """
    return (k + cdim[2] * (j + cdim[1] * i))


def get_ovdengrid(filepath, outpath, size, rank, target_grid_width=2.0, 
                  pad_region=2):
    
    # Define pad region cell numer (double pad_region)
    pad_cells = 2 * pad_region

    # Open HDF5 file
    hdf = h5py.File(filepath, "r")

    # Get metadata
    boxsize = hdf["Header"].attrs["BoxSize"]
    z = hdf["Header"].attrs["Redshift"]
    nparts = hdf["/PartType1/Masses"].size
    pmass = hdf["Header"].attrs["InitialMassTable"][1]
    cdim = hdf["Cells/Meta-data"].attrs["dimension"]
    ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
    cell_width = hdf["Cells/Meta-data"].attrs["size"]
    half_cell_width = cell_width / 2

    # Calculate the mean density
    tot_mass = nparts * pmass
    mean_density = tot_mass / (boxsize[0] * boxsize[1] * boxsize[2])

    # Set up overdensity grid properties
    ovden_cdim = np.int32(cell_width / target_grid_width)
    ovden_cell_width = cell_width / ovden_cdim
    full_grid_ncells = ovden_cdim * cdim
    ovden_cell_volume = (ovden_cell_width[0] * ovden_cell_width[1]
                         * ovden_cell_width[2])

    # Print some fun stuff
    if rank == 0:
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Npart:", nparts)
        print("Particle Mass:", pmass)
        print("Number of simulation cells:", ncells)
        print("Mean Density:", mean_density)
        print("Sim Cell Width:", cell_width)
        print("Grid Cell Width:", ovden_cell_width)
        print("Grid Cell Volume:", ovden_cell_width[0]
              * ovden_cell_width[1] * ovden_cell_width[2])
        print("Cell Grid NCells:", ovden_cdim)
        print("Parent Grid NCells:", full_grid_ncells)

    # Ensure we can wrap correctly without shifting multiple cells
    assert np.all(pad_region < ovden_cdim), "Can't have a pad region " \
                                            "larger than a cell due to " \
                                            "wrapping assumptions"

    # Get the list of simulation cell indices and the associated ijk references
    all_cells = []
    i_s = []
    j_s = []
    k_s = []
    for i in range(cdim[0]):
        for j in range(cdim[1]):
            for k in range(cdim[2]):
                cell = (k + cdim[2] * (j + cdim[1] * i))
                all_cells.append(cell)
                i_s.append(i)
                j_s.append(j)
                k_s.append(k)

    # Find the cells and simulation ijk grid references
    # that this rank has to work on
    rank_cells = np.linspace(0, len(all_cells), size + 1, dtype=int)
    my_cells = all_cells[rank_cells[rank]: rank_cells[rank + 1]]
    my_i_s = i_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_j_s = j_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_k_s = k_s[rank_cells[rank]: rank_cells[rank + 1]]

    print("Rank=", rank, "- My Ncells=", len(my_cells))

    # Open HDF5 file
    hdf_out = h5py.File(outpath, "w")

    # Store some metadata about the parent box
    parent = hdf_out.create_group("Parent")
    parent.attrs["Boxsize"] = boxsize
    parent.attrs["Redshift"] = z
    parent.attrs["Npart"] = nparts
    parent.attrs["Ncells"] = cdim
    parent.attrs["Cell_Width"] = cell_width
    parent.attrs["DM_Mass"] = pmass
    parent.attrs["Mean_Density"] = mean_density

    # Store some metadata about the overdensity grid
    ovden_grid_grp = hdf_out.create_group("Delta_grid")
    ovden_grid_grp.attrs["Cell_Width"] = ovden_cell_width
    ovden_grid_grp.attrs["Ncells_Total"] = full_grid_ncells
    ovden_grid_grp.attrs["Ncells_PerSimCell"] = ovden_cdim
    ovden_grid_grp.attrs["NPadding_Cells"] = pad_region

    # Loop over cells calculating the overdensity grid
    for i, j, k, my_cell in zip(my_i_s, my_j_s, my_k_s, my_cells):

        # Set up array to store this cells overdensity grid
        # with pad region of 1 cell
        mass_grid_this_cell = np.zeros((ovden_cdim[0] + pad_cells,
                                        ovden_cdim[1] + pad_cells,
                                        ovden_cdim[2] + pad_cells))

        # Retrieve the offset and counts for this cell
        my_offset = hdf["/Cells/OffsetsInFile/PartType1"][my_cell]
        my_count = hdf["/Cells/Counts/PartType1"][my_cell]

        # Define the edges of this cell with pad region
        my_edges = hdf["/Cells/Centres"][my_cell, :] - half_cell_width

        if my_count > 0:
            poss = hdf["/PartType1/Coordinates"][
                   my_offset:my_offset + my_count, :] - my_edges \
                   + (pad_region * ovden_cell_width)
            masses = hdf["/PartType1/Masses"][
                     my_offset:my_offset + my_count]

            # Handle the edge case where a particle has been
            # wrapped outside the box
            poss[poss > boxsize] -= boxsize[0]
            poss[poss < 0] += boxsize[0]

            # Compute overdensity grid ijk references
            ovden_ijk = np.int64(poss / ovden_cell_width)

            # Store the mass in each grid cell
            for ii, jj, kk, m in zip(ovden_ijk[:, 0], ovden_ijk[:, 1],
                                     ovden_ijk[:, 2], masses):
                mass_grid_this_cell[ii, jj, kk] += m

            # Convert the mass entries to overdensities
            # (1 + \delta(x) = \rho(x) / \bar{\rho}
            ovden_grid_this_cell = ((mass_grid_this_cell / ovden_cell_volume)
                                    / mean_density)

        else:
            ovden_grid_this_cell = mass_grid_this_cell

        # Create a group for this cell
        this_cell = hdf_out.create_group(str(i) + "_" + str(j)
                                           + "_" + str(k))
        this_cell.attrs["Sim_Cell_Index"] = my_cell
        this_cell.attrs["Sim_Cell_Edges"] = my_edges
        this_cell.create_dataset("grid",
                                 data=ovden_grid_this_cell,
                                 shape=ovden_grid_this_cell.shape,
                                 dtype=ovden_grid_this_cell.dtype,
                                 compression="lzf")

    hdf_out.close()
    hdf.close()


def create_meta_file(metafile, rankfile_dir, outfile_without_rank,
                     size, pad_region):

    # Define padding pixels
    pad_cells = 2 * pad_region
    
    # Change to the data directory to ensure relative paths work
    os.chdir(rankfile_dir)

    # Write the metadata from rank 0 file to meta file
    rank0file = (outfile_without_rank
                 + "rank%s.hdf5" % "0".zfill(4))  # get rank 0 file
    hdf_rank0 = h5py.File(rank0file, "r")  # open rank 0 file
    hdf_meta = h5py.File(metafile, "w")  # create meta file
    for root_key in ["Parent", "Delta_grid"]:  # loop over root groups
        grp = hdf_meta.create_group(root_key)
        for key in hdf_rank0[root_key].attrs.keys():
            grp.attrs[key] = hdf_rank0[root_key].attrs[key]  # write attrs

    hdf_rank0.close()

    # Get the full parent overdensity grid dimensions
    # (this has a 1 cell pad region)
    ngrid_cells = hdf_meta["Delta_grid"].attrs["Ncells_Total"]

    # Get the simulation cell information and grid cdim
    cdim = hdf_meta["Parent"].attrs["Ncells"][0]
    sim_cell_width = hdf_meta["Parent"].attrs["Cell_Width"]
    cdim_per_cell = hdf_meta["Delta_grid"].attrs["Ncells_PerSimCell"]

    # Get the grid cell width
    grid_cell_width = hdf_meta["Delta_grid"].attrs["Cell_Width"]

    # Get padding region in Mpc
    pad_mpc = pad_region * grid_cell_width

    # Set up full grid array
    full_grid = np.zeros((ngrid_cells[0], ngrid_cells[1], ngrid_cells[2]))

    # Loop over rank files creating external links
    for other_rank in range(size):

        # Get the path to this rank
        rankfile = (outfile_without_rank
                    + "rank%s.hdf5" % str(other_rank).zfill(4))

        # Open rankfile
        hdf_rank = h5py.File(rankfile, "r")

        # Loop over groups creating external links with relative path
        # and the full grid
        for key in hdf_rank.keys():

            if key in ["Parent", "Delta_grid"]:
                continue

            # Extract sim cell grid coordinates
            i, j, k = (int(ijk) for ijk in key.split("_"))

            # Get this cell's hdf5 group and edges
            cell_grp = hdf_rank[key]
            edges = cell_grp.attrs["Sim_Cell_Edges"]
            
            # Get the overdensity grid and convert to mass
            grid = cell_grp["grid"][...]
            dimens = grid.shape

            # Get the indices for this cell edge
            # NOTE: These can be negative or larger than the full_grid array
            # but are wrapped later
            ilow = i * cdim_per_cell[0] - pad_region
            jlow = j * cdim_per_cell[1] - pad_region
            klow = k * cdim_per_cell[2] - pad_region
            ihigh = ilow + dimens[0]
            jhigh = jlow + dimens[1]
            khigh = klow + dimens[2]

            print(ilow, ihigh, jlow, jhigh, klow, khigh)

            # If we are not at the edges we don't need any wrapping
            # and can just assign the grid at once
            if (i != 0 and i < cdim - 1
                and j != 0 and j < cdim - 1
                and k != 0 and k < cdim - 1):

                full_grid[ilow: ihigh, jlow: jhigh, klow: khigh] += grid

            else:  # we must wrap

                # Define indices ranges
                irange = np.arange(ilow, ihigh, 1, dtype=int)
                jrange = np.arange(jlow, jhigh, 1, dtype=int)
                krange = np.arange(klow, khigh, 1, dtype=int)

                # To allow for wrapping we need to assign cell by cell ( :( )
                for i_grid, i_full in enumerate(irange):
                    for j_grid, j_full in enumerate(jrange):
                        for k_grid, k_full in enumerate(krange):
                            full_grid[i_full % ngrid_cells[0],
                                      j_full % ngrid_cells[1],
                                      k_full % ngrid_cells[2]] += grid[i_grid,
                                                                      j_grid,
                                                                      k_grid]

        hdf_rank.close()

    hdf_meta.create_dataset("Parent_Grid",
                            data=full_grid,
                            shape=full_grid.shape,
                            dtype=full_grid.dtype, compression="lzf")

    # Loop over cells making groups for each cell in the single file
    for i in range(cdim):
        for j in range(cdim):
            for k in range(cdim):

                # Get cell index and cell edges
                my_cell = (k + cdim * (j + cdim * i))
                my_edges = np.array([i, j, k]) * sim_cell_width

                # Create a group for this cell
                this_cell = hdf_meta.create_group(str(i) + "_" + str(j)
                                                  + "_" + str(k))

                cell_grid = full_grid[i: i + int(cdim_per_cell[0]),
                                      j: j + int(cdim_per_cell[1]),
                                      k: k + int(cdim_per_cell[2])]

                # Write out cell data
                this_cell.attrs["Sim_Cell_Index"] = my_cell
                this_cell.attrs["Sim_Cell_Edges"] = my_edges
                this_cell.create_dataset("grid", data=cell_grid,
                                         shape=cell_grid.shape,
                                         dtype=cell_grid.dtype,
                                         compression="lzf")

    hdf_meta.close()


if __name__ == "__main__":
    
    # Define pad region
    pad_region = 5

    # Get the commandline argument for which snapshot
    num = int(sys.argv[1])

    # Get the simulation "tag"
    sim_tag = sys.argv[2]
    
    # Get the simulation "type"
    sim_type = sys.argv[3]

    # Extract the snapshot string
    snaps = [str(i).zfill(4) for i in range(0, 25)]
    snap = snaps[num]

    # Define input path
    if sim_type == "HYDRO":
        inpath = "/cosma8/data/dp004/flamingo/Runs/" + sim_tag + "/" \
                 "" + sim_type + "_FIDUCIAL/snapshots/flamingo_" + snap \
                 + "/flamingo_" + snap + ".hdf5"
    else:
        inpath = "/cosma8/data/dp004/flamingo/Runs/" + sim_tag + "/" \
                 "" + sim_type + "_FIDUCIAL/flamingo_" + snap \
                 + "/flamingo_" + snap + ".hdf5"

    # Define out file name
    outfile = "overdensity_" + sim_tag + "_" + sim_type + "_" \
              "snap%s_rank%s.hdf5" % (snap, str(rank).zfill(4))
    metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
    outfile_without_rank = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s_" % snap

    # Define output paths
    # For cosma7 (now read only)
    # out_dir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
    #           "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap
    # For cosma8
    out_dir = "/cosma8/data/dp004/dc-rope1/FLARES-2/Parent/" \
              "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap
    if not os.path.isdir(out_dir) and rank == 0:
        os.mkdir(out_dir)
    outpath = out_dir + "/" + outfile  # Combine file and path
    ini_rankpath = out_dir + "/" + outfile_without_rank  # rankless string

    # Get the overdensity grid for this rank
    get_ovdengrid(inpath, outpath, size, rank, target_grid_width=2.0,
                  pad_region=pad_region)

    # Ensure all files are finished writing
    comm.Barrier()

    # Create the meta file now we have each individual rank file
    if rank == 0:
        create_meta_file(metafile, out_dir, outfile_without_rank, size,
                         pad_region=pad_region)
