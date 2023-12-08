import gc
import sys

import h5py
import numpy as np
from mpi4py import MPI

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def get_smoothed_grid(snap, ini_kernel_width, outdir, rank, size):
    # Get the simulation "tag"
    sim_tag = sys.argv[2]

    # Get the simulation "type"
    sim_type = sys.argv[3]

    # Define path to file
    metafile = "overdensity_" + sim_tag + "_" + sim_type + "_snap%s.hdf5" % snap
    path = outdir + metafile

    # Open file
    hdf = h5py.File(path, "r")

    # Get metadata
    boxsize = hdf["Parent"].attrs["Boxsize"]
    mean_density = hdf["Parent"].attrs["Mean_Density"]
    ngrid_cells = hdf["Delta_grid"].attrs["Ncells_Total"]
    grid_cell_width = hdf["Delta_grid"].attrs["Cell_Width"]

    # Compute actual kernel width 
    cells_per_kernel = np.int32(np.ceil(ini_kernel_width / grid_cell_width[0]))
    kernel_width = cells_per_kernel * grid_cell_width

    # Print some nice things
    if rank == 0:
        print("Boxsize:", boxsize)
        print("Mean Density:", mean_density)
        print("Grid Cell Width:", grid_cell_width)
        print("Kernel Width:", kernel_width)
        print("Grid cells in kernel:", cells_per_kernel)
        print("Grid cells total:", ngrid_cells)

    # Get full parent grid
    ovden_grid = hdf["Parent_Grid"]

    # Get results array dimensions
    grid_shape = (ovden_grid.shape[0] - cells_per_kernel,
                  ovden_grid.shape[1] - cells_per_kernel,
                  ovden_grid.shape[2] - cells_per_kernel)
    full_nregion_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]

    if rank == 0:
        print("Number of regions:", full_nregion_cells)

    # Find the cells and simulation ijk grid references
    # that this rank has to work on
    rank_cells = np.linspace(0, full_nregion_cells, size + 1, dtype=int)

    print("Rank: %d has %d cells" % (rank,
                                     rank_cells[rank + 1] - rank_cells[rank]))

    # Set up arrays to store this ranks results and the indices
    # in the full array
    region_vals = np.zeros(rank_cells[rank + 1] - rank_cells[rank])
    region_stds = np.zeros(region_vals.size)
    region_inds = np.zeros(region_vals.size, dtype=int)
    centres = np.zeros((region_vals.size, 3))

    # Find the minimum and maximum i, j and k to slice main grid
    min_i, min_j, min_k = np.inf, np.inf, np.inf
    max_i, max_j, max_k = 0, 0, 0
    for cid in range(rank_cells[rank], rank_cells[rank + 1]):
        i = int(cid / (grid_shape[1] * grid_shape[2]))
        j = int((cid / grid_shape[2]) % grid_shape[1])
        k = int(cid % grid_shape[2])
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

    print("Rank=%d has cells in range [%d-%d, %d-%d, %d-%d]"
          % (rank, min_i, max_i, min_j, max_j, min_k, max_k))

    # Get only the cells of the grid this rank needs
    ovden_grid = ovden_grid[
                 min_i: max_i + cells_per_kernel,
                 min_j: max_j + cells_per_kernel,
                 min_k: max_k + cells_per_kernel]

    hdf.close()

    # Loop over the region kernels
    ind = 0
    for cid in range(rank_cells[rank], rank_cells[rank + 1]):
        # Get the upper and lower grid coordinates for this rank
        i = int(cid / (grid_shape[1] * grid_shape[2]))
        j = int((cid / grid_shape[2]) % grid_shape[1])
        k = int(cid % grid_shape[2])

        # Shift is for this ranks subsample of the grid
        low_i = i - min_i
        low_j = j - min_j
        low_k = k - min_k

        # Get the index for this smoothed grid cell in the full grid
        full_ind = (k + grid_shape[2] * (j + grid_shape[1] * i))

        # Get the mean of these overdensities
        ovden_kernel = np.mean(ovden_grid[low_i: low_i
                                                 + cells_per_kernel,
                               low_j: low_j
                                      + cells_per_kernel,
                               low_k: low_k
                                      + cells_per_kernel])

        # Get the standard deviation of this region
        ovden_kernel_std = np.std(ovden_grid[
                                  low_i: low_i + cells_per_kernel,
                                  low_j: low_j + cells_per_kernel,
                                  low_k: low_k + cells_per_kernel])

        # Store edges
        edges = np.array([i * grid_cell_width[0],
                          j * grid_cell_width[1],
                          k * grid_cell_width[2]])
        centres[ind, :] = edges + (kernel_width / 2)

        # Store the results and index
        region_vals[ind] = ovden_kernel
        region_stds[ind] = ovden_kernel_std
        region_inds[ind] = full_ind

        ind += 1

    # Set up the outpath for each rank file
    outpath = outdir + "smoothed_" + metafile.split(".")[0] \
              + "_kernel%d_rank%d.hdf5" % (ini_kernel_width, rank)

    # Write out the results of smoothing
    hdf = h5py.File(outpath, "w")
    hdf.attrs["Kernel_Width"] = kernel_width
    hdf.create_dataset("Region_Overdensity", data=region_vals,
                       shape=region_vals.shape, dtype=region_vals.dtype,
                       compression="lzf")
    hdf.create_dataset("Region_Overdensity_Stdev", data=region_stds,
                       shape=region_stds.shape, dtype=region_stds.dtype,
                       compression="lzf")
    hdf.create_dataset("Region_Centres", data=centres,
                       shape=centres.shape, dtype=centres.dtype,
                       compression="lzf")
    hdf.create_dataset("Region_Indices", data=region_inds,
                       shape=region_inds.shape, dtype=region_inds.dtype,
                       compression="lzf")

    hdf_grid = h5py.File(path, "r")

    for root_key in ["Parent", "Delta_grid"]:
        for key in hdf_grid[root_key].attrs.keys():
            hdf.attrs[root_key + "_" + key] = hdf_grid[root_key].attrs[key]

    hdf_grid.close()

    hdf.close()

    # Clean up
    del region_vals, region_stds, region_inds, centres, ovden_grid
    gc.collect()

    comm.Barrier()

    if rank == 0:

        #  ========== Define arrays to store the collected results ==========

        # Set up arrays for the smoothed grid
        final_region_vals = np.zeros(full_nregion_cells)
        final_region_stds = np.zeros(final_region_vals.size)

        # Set up array to store centres
        final_centres = np.zeros((final_region_vals.size, 3))

        # Set up the outpaths
        outpath = outdir + "smoothed_" + metafile.split(".")[0] \
                  + "_kernel%d.hdf5" % ini_kernel_width
        outpath0 = outdir + "smoothed_" + metafile.split(".")[0] \
                   + "_kernel%d_rank0.hdf5" % ini_kernel_width

        # Open file to combine results
        hdf = h5py.File(outpath, "w")

        # Open rank 0 file to get metadata
        hdf_rank0 = h5py.File(outpath0, "r")  # open rank 0 file
        for key in hdf_rank0.attrs.keys():
            hdf.attrs[key] = hdf_rank0.attrs[key]  # write attrs

        hdf_rank0.close()

        for other_rank in range(size):
            # Set up the outpath for each rank file
            rank_outpath = outdir + "smoothed_" + metafile.split(".")[0] \
                           + "_kernel%d_rank%d.hdf5" % (ini_kernel_width,
                                                        other_rank)

            hdf_rank = h5py.File(rank_outpath, "r")  # open rank 0 file

            # Get indices
            inds = hdf_rank["Region_Indices"][...]

            # Combine this rank's results into the final array
            ods = hdf_rank["Region_Overdensity"][...]
            final_region_vals[inds] += ods

            final_region_stds[inds] += hdf_rank["Region_Overdensity_Stdev"][
                ...]
            final_centres[inds] += hdf_rank["Region_Centres"][...]

            hdf_rank.close()

        sinds = np.argsort(final_region_vals)

        hdf.create_dataset("Sorted_Indices",
                           data=sinds,
                           shape=sinds.shape,
                           dtype=sinds.dtype,
                           compression="lzf")
        hdf.create_dataset("Region_Overdensity",
                           data=final_region_vals,
                           shape=final_region_vals.shape,
                           dtype=final_region_vals.dtype,
                           compression="lzf")
        hdf.create_dataset("Region_Overdensity_Stdev",
                           data=final_region_stds,
                           shape=final_region_stds.shape,
                           dtype=final_region_stds.dtype,
                           compression="lzf")
        hdf.create_dataset("Region_Centres",
                           data=final_centres,
                           shape=final_centres.shape,
                           dtype=final_centres.dtype,
                           compression="lzf")

        hdf.close()


if __name__ == "__main__":
    # Get the commandline argument for which snapshot
    num = int(sys.argv[1])

    # Get the simulation "tag"
    sim_tag = sys.argv[2]

    # Get the simulation "type"
    sim_type = sys.argv[3]

    # Extract the snapshot string
    snaps = [str(i).zfill(4) for i in range(0, 23)]
    snap = snaps[num]

    # Define output paths
    # For cosma7 (now read only)
    # out_dir = "/cosma7/data/dp004/FLARES/FLARES-2/Parent/" \
    #           "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap
    # For cosma8
    out_dir = "/cosma8/data/dp004/dc-rope1/FLARES-2/Parent/" \
              "overdensity_gridding/" + sim_tag + "/" + sim_type + "/snap_" + snap + "/"

    # Run smoothing
    ini_kernel_width = 25  # in cMpc
    get_smoothed_grid(snap, ini_kernel_width, out_dir, rank, size)
