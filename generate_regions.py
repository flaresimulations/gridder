"""
Create a 3D overdensity grid of a parent simulation.

This script generates a 3D overdensity grid of a SWIFT simulation by using the
RegionGenerator class from the region_smoother module. It supports various kernel
diameters for a fixed set of grid points.

Example usage:

    mpirun -np 4 generate_regions.py --input /IN/snapshot.hdf5 --output /OUT/grid.hdf5

Command Line Arguments:
    --input (str):
        The SWIFT snapshot to grid.
    --output (str):
        The file path for the grid file.
    --grid_width (float):
        The distance between grid points in simulation units. Default is 2.
    --kernel_diameters (List[float]):
        The diameter of the spherical region kernel in simulation units. Required.
    --delete_distributed (int):
        Should the distributed files be deleted after combination? Default is 0.
    --nthreads (int):
        The number of threads to use within a rank. Default is 1.

Usage:
    python generate_regions.py --input /path/to/snapshot.hdf5 --output /path/to/grid.hdf5
                               --grid_width 2 --kernel_diameters 2.0 3.0 4.0
                               --delete_distributed 0 --nthreads 4
"""

import time
import argparse
from mpi4py import MPI
from region_smoother import RegionGenerator


def main():
    """
    Main function for generating a 3D overdensity grid of a SWIFT simulation.
    """
    total_start = time.time()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.rank  # rank of this process

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Get a 3D overdensity grid of a SWIFT simulation."
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
        "--grid_width",
        type=float,
        help="The distance between grid points in simulation units",
        default=2,
    )
    parser.add_argument(
        "--kernel_diameters",
        nargs="+",
        type=float,
        help="The diameter of the spherical region kernel in simulation units",
        required=True,
    )
    parser.add_argument(
        "--delete_distributed",
        type=int,
        help="Should the distributed files be deleted after combination?",
        default=0,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use within a rank",
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="How many SWIFT cells to read and process at a time. "
        "If negative all cells on a rank will be done in a single batch",
        default=100,
    )
    args = parser.parse_args()

    # Get the grid for each kernel width passed on the command line
    # It would be nice to maintain the domain decomp and only regrid
    # but the domain decomp is dependant on the kernel radius and
    # using the largest would be wasteful!
    for kernel_width in args.kernel_diameters:
        if rank == 0:
            print()
            print(f"+++++++++++ Gridding for {kernel_width:.2f} kernel +++++++++++")
            print()

        # Create the grid instance
        gridder = RegionGenerator(
            args.input,
            args.output,
            args.grid_width,
            kernel_width=kernel_width,
            nthreads=args.nthreads,
        )

        # Decompose the grid cells of nranks and construct the KDTrees
        start = time.time()
        gridder.domain_decomp()
        if rank == 0:
            print(f"Decomposing the grid took {time.time() - start} seconds")

        # Perform the gridding
        start = time.time()
        gridder.get_grid()
        comm.Barrier()
        if rank == 0:
            print(f"Computing overdensities took {time.time() - start} seconds")
            start = time.time()
            gridder.combine_distributed_files(
                delete_distributed=args.delete_distributed,
            )
            print(f"Combining grid files took {time.time() - start} seconds")

            print(f"Entire process took {time.time() - total_start} seconds")


if __name__ == "__main__":
    main()
