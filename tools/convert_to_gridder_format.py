#!/usr/bin/env python3
"""
Convert HDF5 simulation snapshots to gridder-compatible format.

This script converts HDF5 files with arbitrary key names to the format
expected by the parent_gridder code. It supports both serial and MPI
execution for processing large files efficiently.

In MPI mode, each rank writes a separate file and a virtual HDF5 file
is created to provide a unified view of the data.
"""

import argparse
import sys
import numpy as np
import h5py

# Try to import mpi4py for MPI support
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 snapshot to gridder format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serial conversion
  python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key Coordinates --masses-key Masses

  # MPI conversion (creates output_rank_*.hdf5 files + virtual file)
  mpirun -np 4 python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key PartType1/Coordinates --masses-key PartType1/Masses

  # With custom particle type prefix
  python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key MyCoords --masses-key MyMasses \\
      --particle-type PartType1
        """
    )

    parser.add_argument(
        "input_file",
        help="Input HDF5 file path"
    )

    parser.add_argument(
        "output_file",
        help="Output HDF5 file path (or base name for MPI mode)"
    )

    parser.add_argument(
        "--coordinates-key",
        required=True,
        help="HDF5 key for particle coordinates in input file"
    )

    parser.add_argument(
        "--masses-key",
        required=True,
        help="HDF5 key for particle masses in input file"
    )

    parser.add_argument(
        "--particle-type",
        default="PartType1",
        help="Particle type group name in output file (default: PartType1)"
    )

    parser.add_argument(
        "--copy-header",
        action="store_true",
        help="Copy Header group from input to output"
    )

    parser.add_argument(
        "--header-key",
        default="Header",
        help="HDF5 key for header group in input file (default: Header)"
    )

    return parser.parse_args()


def get_mpi_info():
    """Get MPI rank and size, or return defaults if not using MPI."""
    if HAS_MPI and MPI.COMM_WORLD.size > 1:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        using_mpi = True
    else:
        comm = None
        rank = 0
        size = 1
        using_mpi = False

    return comm, rank, size, using_mpi


def get_particle_count(input_file, masses_key, rank, size):
    """
    Get total particle count and determine chunk for this rank.

    Returns:
        total_particles: Total number of particles in input file
        start_idx: Starting index for this rank
        count: Number of particles for this rank to process
    """
    with h5py.File(input_file, 'r') as f:
        if masses_key not in f:
            raise KeyError(f"Masses key '{masses_key}' not found in input file")

        total_particles = f[masses_key].shape[0]

    # Divide particles among ranks
    particles_per_rank = total_particles // size
    remainder = total_particles % size

    if rank < remainder:
        start_idx = rank * (particles_per_rank + 1)
        count = particles_per_rank + 1
    else:
        start_idx = rank * particles_per_rank + remainder
        count = particles_per_rank

    return total_particles, start_idx, count


def convert_file_serial(args):
    """Convert file in serial mode (single output file)."""
    print(f"Converting {args.input_file} -> {args.output_file}")

    with h5py.File(args.input_file, 'r') as f_in:
        # Check input keys exist
        if args.coordinates_key not in f_in:
            raise KeyError(f"Coordinates key '{args.coordinates_key}' not found")
        if args.masses_key not in f_in:
            raise KeyError(f"Masses key '{args.masses_key}' not found")

        coords = f_in[args.coordinates_key][:]
        masses = f_in[args.masses_key][:]

        npart = masses.shape[0]

        # Verify shapes
        if coords.shape[0] != npart:
            raise ValueError(
                f"Shape mismatch: coordinates has {coords.shape[0]} particles, "
                f"masses has {npart}"
            )

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Coordinates must be shape (N, 3), got {coords.shape}"
            )

        print(f"  Found {npart} particles")
        print(f"  Coordinates shape: {coords.shape}")
        print(f"  Masses shape: {masses.shape}")

        # Create output file
        with h5py.File(args.output_file, 'w') as f_out:
            # Create particle type group
            pt_group = f_out.create_group(args.particle_type)

            # Write coordinates and masses
            pt_group.create_dataset(
                'Coordinates',
                data=coords,
                compression='gzip',
                compression_opts=4
            )
            pt_group.create_dataset(
                'Masses',
                data=masses,
                compression='gzip',
                compression_opts=4
            )

            # Copy header if requested
            if args.copy_header and args.header_key in f_in:
                print(f"  Copying header from {args.header_key}")
                f_in.copy(args.header_key, f_out, 'Header')

            print(f"✓ Conversion complete: {args.output_file}")


def convert_file_mpi(args, comm, rank, size):
    """Convert file in MPI mode (one file per rank + virtual file)."""
    # Generate output filenames
    base_name = args.output_file.replace('.hdf5', '')
    rank_file = f"{base_name}_rank_{rank}.hdf5"
    virtual_file = f"{base_name}.hdf5"

    # Get particle distribution
    total_particles, start_idx, count = get_particle_count(
        args.input_file, args.masses_key, rank, size
    )

    if rank == 0:
        print(f"Converting {args.input_file} -> {base_name}_rank_*.hdf5")
        print(f"  Total particles: {total_particles}")
        print(f"  MPI ranks: {size}")
        print(f"  Particles per rank: ~{total_particles // size}")

    comm.Barrier()

    # Each rank reads and writes its chunk
    print(f"Rank {rank}: Processing particles {start_idx} to {start_idx + count - 1}")

    with h5py.File(args.input_file, 'r') as f_in:
        # Read this rank's chunk
        coords = f_in[args.coordinates_key][start_idx:start_idx + count]
        masses = f_in[args.masses_key][start_idx:start_idx + count]

        # Validate shapes (same as serial mode)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Rank {rank}: Coordinates must be Nx3 array, got shape {coords.shape}"
            )
        if masses.ndim != 1 or len(masses) != coords.shape[0]:
            raise ValueError(
                f"Rank {rank}: Masses shape {masses.shape} doesn't match "
                f"coordinates shape {coords.shape}"
            )

        # Write to rank-specific file
        with h5py.File(rank_file, 'w') as f_out:
            pt_group = f_out.create_group(args.particle_type)

            pt_group.create_dataset(
                'Coordinates',
                data=coords,
                compression='gzip',
                compression_opts=4
            )
            pt_group.create_dataset(
                'Masses',
                data=masses,
                compression='gzip',
                compression_opts=4
            )

            # Copy header to first rank's file
            if rank == 0 and args.copy_header and args.header_key in f_in:
                f_in.copy(args.header_key, f_out, 'Header')

        print(f"Rank {rank}: Wrote {count} particles to {rank_file}")

    comm.Barrier()

    # Rank 0 creates virtual dataset
    if rank == 0:
        print(f"Rank 0: Creating virtual file {virtual_file}")
        create_virtual_file(
            base_name, size, total_particles, args.particle_type,
            args.copy_header
        )
        print(f"✓ Conversion complete: {virtual_file}")


def create_virtual_file(base_name, nranks, total_particles, particle_type,
                        include_header):
    """
    Create a virtual HDF5 file that combines all rank files.

    Args:
        base_name: Base output filename (without .hdf5)
        nranks: Number of MPI ranks
        total_particles: Total number of particles across all ranks
        particle_type: Particle type group name
        include_header: Whether to include header from rank 0
    """
    virtual_file = f"{base_name}.hdf5"

    # Infer dtype from first rank file for consistency
    rank_0_file = f"{base_name}_rank_0.hdf5"
    try:
        with h5py.File(rank_0_file, 'r') as f:
            coord_dtype = f[f'{particle_type}/Coordinates'].dtype
            mass_dtype = f[f'{particle_type}/Masses'].dtype
            # Use float64 if dtypes differ or as fallback
            dtype = coord_dtype if coord_dtype == mass_dtype else np.float64
    except Exception as e:
        print(f"Warning: Could not infer dtype from {rank_0_file}: {e}")
        print(f"Falling back to np.float64")
        dtype = np.float64

    # Create virtual layout for coordinates (N, 3)
    coord_layout = h5py.VirtualLayout(
        shape=(total_particles, 3),
        dtype=dtype
    )

    # Create virtual layout for masses (N,)
    mass_layout = h5py.VirtualLayout(
        shape=(total_particles,),
        dtype=dtype
    )

    # Get particle counts from each rank file
    particle_counts = []
    for i in range(nranks):
        rank_file = f"{base_name}_rank_{i}.hdf5"
        with h5py.File(rank_file, 'r') as f:
            npart = f[f'{particle_type}/Masses'].shape[0]
            particle_counts.append(npart)

    # Map each rank's data into the virtual datasets
    offset = 0
    for i in range(nranks):
        rank_file = f"{base_name}_rank_{i}.hdf5"
        count = particle_counts[i]

        # Create virtual source for this rank
        coord_vsource = h5py.VirtualSource(
            rank_file,
            f'{particle_type}/Coordinates',
            shape=(count, 3)
        )
        mass_vsource = h5py.VirtualSource(
            rank_file,
            f'{particle_type}/Masses',
            shape=(count,)
        )

        # Map into virtual layout
        coord_layout[offset:offset + count, :] = coord_vsource
        mass_layout[offset:offset + count] = mass_vsource

        offset += count

    # Create virtual file
    with h5py.File(virtual_file, 'w', libver='latest') as f:
        pt_group = f.create_group(particle_type)

        # Create virtual datasets
        pt_group.create_virtual_dataset('Coordinates', coord_layout)
        pt_group.create_virtual_dataset('Masses', mass_layout)

        # Copy header from rank 0 file if requested
        if include_header:
            rank0_file = f"{base_name}_rank_0.hdf5"
            with h5py.File(rank0_file, 'r') as f0:
                if 'Header' in f0:
                    f0.copy('Header', f, 'Header')


def main():
    """Main entry point."""
    args = parse_args()

    # Get MPI info
    comm, rank, size, using_mpi = get_mpi_info()

    # Print info from rank 0
    if rank == 0:
        print("=" * 60)
        print("HDF5 to Gridder Format Converter")
        print("=" * 60)
        if using_mpi:
            print(f"Mode: MPI ({size} ranks)")
        else:
            print("Mode: Serial")
        print()

    try:
        if using_mpi:
            convert_file_mpi(args, comm, rank, size)
        else:
            convert_file_serial(args)

        if rank == 0:
            print()
            print("=" * 60)
            print("Conversion successful!")
            print("=" * 60)

        return 0

    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

        if using_mpi:
            comm.Abort(1)

        return 1


if __name__ == "__main__":
    sys.exit(main())
