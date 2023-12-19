""" Create a 3D overdensity grid of a parent simulation.

This can do any number of kernel diameters for any fixed set of grid points.

Example usage:

    mpirun -np 4 generate_regions.py --input /IN/snapshot.hdf5 --output /OUT/grid.hdf5
"""
import time
import argparse
from mpi4py import MPI
from region_smoother import RegionGenerator


def main():
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
