""" Create a 3D mass histogram of the parent simulation.

Example usage:

    mpirun -np 4 grid_parent.py --input /IN/PATH/snapshot.hdf5 --output /GRID/PATH/grid.hdf5
"""
import argparse
from mpi4py import MPI
from gridder import GridGenerator


def main():
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.rank  # rank of this process

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Get a 3D mass histogram of a SWIFT simulation."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="The SWIFT snapshot to grid",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The file path for the grid file",
    )
    parser.add_argument(
        "--cell_width",
        type=float,
        help="The width of a grid cell in Mpc",
        default=2,
    )
    args = parser.parse_args()

    # Create the grid instance
    gridder = GridGenerator(args.input, args.output, args.cell_width)

    # Decompose the simulation cells into slices
    gridder.domain_decomp()

    # Perform the gridding
    gridder.get_grid()

    comm.Barrier()
    if rank == 0:
        gridder.combine_distributed_files(delete_distributed=False)


if __name__ == "__main__":
    main()
