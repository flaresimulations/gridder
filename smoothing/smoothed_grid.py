""" Smooth the FLARES-2 parent grid using the GridSmoother.

These grids come from FLAMINGO. Before running this script grid_parent must
have been run to generate the grid files.

Example usage:

    mpirun -np 4 smoothed_grid.py --grid_file /GRID/PATH/grid.hdf5
"""
import argparse
from mpi4py import MPI
from grid_smoother import GridSmoother


def main():
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    nranks = comm.size  # total number of processes
    rank = comm.rank  # rank of this process

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Apply GridSmoother to a gridded parent simulation."
    )
    parser.add_argument(
        "--grid_file",
        type=str,
        help="The file path of the combined grid file",
    )
    parser.add_argument(
        "--kernel_width",
        type=float,
        help="Kernel diameter in Mpc",
        default=30,
    )
    args = parser.parse_args()

    # Define the output path
    split_grid_path = args.grid_file.split("/")
    out_dir = "/".join(split_grid_path[:-1])

    # Define path to file
    metafile = split_grid_path[-1]

    outpath = (
        out_dir
        + "smoothed_"
        + metafile.split(".")[0]
        + f"_kernel{args.kernel_width}.hdf5"
    )

    # Create the grid instance
    smoother = GridSmoother(args.kernel_width, out_dir + metafile)

    # Decompose the grid
    smoother.decomp_cells()

    # Compute the smoothed grid
    smoother.smooth_grid_cells()

    # Output the grid from each rank to the distributed files
    smoother.write_smoothed_grid_rankfile(outpath)

    comm.Barrier()

    # Convert the distributed files into a single file
    smoother.combine_distributed_files(outpath, metafile, delete_distributed=False)


if __name__ == "__main__":
    main()
