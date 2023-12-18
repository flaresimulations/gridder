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
    parser.add_argument(
        "--delete_distributed",
        type=int,
        help="Should the distributed files be deleted after combination?",
        default=0,
    )
    args = parser.parse_args()

    smoother = GridSmoother(args.kernel_width, args.grid_file)

    # Decompose the grid
    smoother.decomp_cells()
    if rank == 0:
        print("Decomposed cells...")

    # Compute the smoothed grid
    if rank == 0:
        print("Applying the spherical top hat kernel")
    smoother.smooth_grid_cells()

    # Output the grid from each rank to the distributed files
    if rank == 0:
        print("Writing distributed files...")
    smoother.write_smoothed_grid_rankfile()

    comm.Barrier()

    # Convert the distributed files into a single file
    if rank == 0:
        print("Combining files...")
    smoother.combine_distributed_files(
        delete_distributed=args.delete_distributed,
    )


if __name__ == "__main__":
    main()
