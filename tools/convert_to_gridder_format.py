#!/usr/bin/env python3
"""
Convert HDF5 simulation snapshots to gridder-compatible format.

This script converts HDF5 files with arbitrary key names to the format
expected by the parent_gridder code. The conversion includes:

1. Reading particle coordinates and masses from arbitrary HDF5 keys
2. Creating a spatial cell structure for efficient particle lookup
3. Sorting particles by cell index
4. Writing output in the standardized gridder format

The gridder requires a hierarchical cell structure for spatial indexing.
This script creates a regular grid of cells (default: 16x16x16 = 4096 cells)
and assigns each particle to a cell based on its position. Particles are
then sorted by cell index before being written to the output file.

Output HDF5 Structure:
  /Header                              # Simulation metadata (copied if --copy-header)
  /PartType1
    /Coordinates                       # Particle positions (sorted by cell)
    /Masses                            # Particle masses (sorted by cell)
  /Cells
    /Meta-data
      dimension: [cdim, cdim, cdim]    # Number of cells per dimension
      size: [dx, dy, dz]               # Physical size of each cell
    /Counts
      /PartType1                       # Number of particles in each cell
    /OffsetsInFile
      /PartType1                       # Starting index for particles in each cell

Supports both serial and MPI execution for processing large files efficiently.
In MPI mode, each rank writes a separate file and a virtual HDF5 file is
created to provide a unified view of the data.
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
  # Serial conversion (BoxSize read from Header)
  python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key Coordinates --masses-key Masses \\
      --copy-header

  # MPI conversion (creates output_rank_*.hdf5 files + virtual file)
  mpirun -np 4 python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key PartType1/Coordinates --masses-key PartType1/Masses \\
      --copy-header

  # Specify BoxSize and cell dimension manually
  python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key MyCoords --masses-key MyMasses \\
      --boxsize 100.0 100.0 100.0 --cdim 32

  # With custom particle type prefix and finer cell grid
  python convert_to_gridder_format.py input.hdf5 output.hdf5 \\
      --coordinates-key DarkMatter/Positions --masses-key DarkMatter/Masses \\
      --particle-type PartType1 --cdim 64 --copy-header
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

    parser.add_argument(
        "--cdim",
        type=int,
        default=16,
        help="Number of cells per dimension for spatial indexing (default: 16)"
    )

    parser.add_argument(
        "--boxsize",
        type=float,
        nargs=3,
        help="Box size [X, Y, Z] in same units as coordinates. If not provided, will try to read from Header/BoxSize"
    )

    return parser.parse_args()


def write_minimal_header(h5file, boxsize, npart, redshift=0.0, header_name="Header"):
    """Write a minimal Header group expected by the gridder."""
    hdr = h5file.require_group(header_name)
    hdr.attrs["BoxSize"] = np.array(boxsize, dtype=np.float64)

    numpart = np.zeros(6, dtype=np.uint64)
    numpart[1] = npart  # PartType1 slot
    hdr.attrs["NumPart_Total"] = numpart
    hdr.attrs["Redshift"] = np.float64(redshift)


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


def get_boxsize(input_file, args, rank=0):
    """
    Get box size from command line args or Header in input file.

    Returns:
        boxsize: numpy array [X, Y, Z] box dimensions
    """
    if args.boxsize is not None:
        boxsize = np.array(args.boxsize, dtype=np.float64)
        if rank == 0:
            print(f"  Using provided BoxSize: {boxsize}")
        return boxsize

    # Try to read from Header
    with h5py.File(input_file, 'r') as f:
        if args.header_key in f and 'BoxSize' in f[args.header_key].attrs:
            boxsize = np.array(f[args.header_key].attrs['BoxSize'], dtype=np.float64)
            if boxsize.shape == ():  # Scalar -> cubic box
                boxsize = np.array([boxsize, boxsize, boxsize], dtype=np.float64)
            if rank == 0:
                print(f"  Read BoxSize from {args.header_key}/BoxSize: {boxsize}")
            return boxsize

    raise ValueError(
        "BoxSize not found in input file and not provided via --boxsize. "
        "Please specify --boxsize X Y Z or ensure Header/BoxSize exists in input file."
    )


def create_cell_structure(coords, masses, boxsize, cdim):
    """
    Create cell structure for spatial indexing.

    This function:
    1. Assigns each particle to a cell based on position
    2. Counts particles per cell
    3. Sorts particles by cell index
    4. Computes cumulative offsets

    Args:
        coords: Particle coordinates (N, 3)
        masses: Particle masses (N,)
        boxsize: Box dimensions [X, Y, Z]
        cdim: Number of cells per dimension

    Returns:
        sorted_coords: Coordinates sorted by cell index
        sorted_masses: Masses sorted by cell index
        cell_counts: Number of particles in each cell
        cell_offsets: Starting index for each cell
        cell_size: Physical size of cells [X, Y, Z]
    """
    npart = coords.shape[0]
    ncells = cdim ** 3

    # Calculate cell size
    cell_size = boxsize / cdim

    # Assign particles to cells
    # cell_id = k + j*cdim + i*cdim*cdim (row-major order)
    i = np.floor(coords[:, 0] / cell_size[0]).astype(np.int32)
    j = np.floor(coords[:, 1] / cell_size[1]).astype(np.int32)
    k = np.floor(coords[:, 2] / cell_size[2]).astype(np.int32)

    # Clamp to cell bounds (handle particles exactly at box edge)
    i = np.clip(i, 0, cdim - 1)
    j = np.clip(j, 0, cdim - 1)
    k = np.clip(k, 0, cdim - 1)

    # Compute cell index
    cell_indices = k + j * cdim + i * cdim * cdim

    # Count particles per cell
    cell_counts = np.bincount(cell_indices, minlength=ncells).astype(np.int32)

    # Get sorting indices
    sort_idx = np.argsort(cell_indices)

    # Sort particles by cell
    sorted_coords = coords[sort_idx]
    sorted_masses = masses[sort_idx]

    # Compute cumulative offsets
    cell_offsets = np.zeros(ncells, dtype=np.int64)
    cell_offsets[1:] = np.cumsum(cell_counts[:-1])

    return sorted_coords, sorted_masses, cell_counts, cell_offsets, cell_size


def write_cell_structure(f_out, cell_counts, cell_offsets, cdim, cell_size):
    """
    Write cell structure to HDF5 file.

    Args:
        f_out: Output HDF5 file handle
        cell_counts: Particle counts per cell
        cell_offsets: Cumulative offsets per cell
        cdim: Number of cells per dimension
        cell_size: Physical size of cells [X, Y, Z]
    """
    # Create Cells group
    cells_group = f_out.create_group('Cells')

    # Create Counts subgroup
    counts_group = cells_group.create_group('Counts')
    counts_group.create_dataset('PartType1', data=cell_counts, compression='gzip', compression_opts=4)

    # Create OffsetsInFile subgroup
    offsets_group = cells_group.create_group('OffsetsInFile')
    offsets_group.create_dataset('PartType1', data=cell_offsets, compression='gzip', compression_opts=4)

    # Create Meta-data subgroup
    metadata_group = cells_group.create_group('Meta-data')
    metadata_group.attrs['dimension'] = np.array([cdim, cdim, cdim], dtype=np.int32)
    metadata_group.attrs['size'] = cell_size


def convert_file_serial(args):
    """Convert file in serial mode (single output file)."""
    print(f"Converting {args.input_file} -> {args.output_file}")

    # Get box size
    boxsize = get_boxsize(args.input_file, args)

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

        # Create cell structure
        print(f"  Creating cell structure (cdim={args.cdim})...")
        sorted_coords, sorted_masses, cell_counts, cell_offsets, cell_size = \
            create_cell_structure(coords, masses, boxsize, args.cdim)

        ncells = args.cdim ** 3
        non_empty = np.count_nonzero(cell_counts)
        print(f"  Cells: {ncells} total, {non_empty} non-empty")
        print(f"  Cell size: {cell_size}")

        # Create output file
        with h5py.File(args.output_file, 'w') as f_out:
            # Create particle type group with sorted data
            pt_group = f_out.create_group(args.particle_type)

            # Write sorted coordinates and masses
            pt_group.create_dataset(
                'Coordinates',
                data=sorted_coords,
                compression='gzip',
                compression_opts=4
            )
            pt_group.create_dataset(
                'Masses',
                data=sorted_masses,
                compression='gzip',
                compression_opts=4
            )

            # Write cell structure
            write_cell_structure(f_out, cell_counts, cell_offsets, args.cdim, cell_size)

            # Copy header if requested, otherwise synthesize when boxsize is provided
            if args.copy_header and args.header_key in f_in:
                print(f"  Copying header from {args.header_key}")
                f_in.copy(args.header_key, f_out, 'Header')
            elif args.boxsize is not None:
                print("  Writing minimal Header from provided BoxSize")
                write_minimal_header(f_out, boxsize, npart)

            print(f"✓ Conversion complete: {args.output_file}")


def convert_file_mpi(args, comm, rank, size):
    """
    Convert file in MPI mode (one file per rank + virtual file).

    Note: In MPI mode, particles are sorted by cell within each rank's file,
    but not globally across all ranks. This is acceptable for the gridder
    which can handle per-rank cell structures.
    """
    # Generate output filenames
    base_name = args.output_file.replace('.hdf5', '')
    rank_file = f"{base_name}_rank_{rank}.hdf5"
    virtual_file = f"{base_name}.hdf5"

    # Get particle distribution
    total_particles, start_idx, count = get_particle_count(
        args.input_file, args.masses_key, rank, size
    )

    # Get box size (all ranks need this)
    boxsize = get_boxsize(args.input_file, args, rank)

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

        # Create cell structure for this rank's particles
        print(f"Rank {rank}: Creating cell structure (cdim={args.cdim})...")
        sorted_coords, sorted_masses, cell_counts, cell_offsets, cell_size = \
            create_cell_structure(coords, masses, boxsize, args.cdim)

        ncells = args.cdim ** 3
        non_empty = np.count_nonzero(cell_counts)
        print(f"Rank {rank}: Cells: {ncells} total, {non_empty} non-empty in this rank")

        # Write to rank-specific file
        with h5py.File(rank_file, 'w') as f_out:
            pt_group = f_out.create_group(args.particle_type)

            pt_group.create_dataset(
                'Coordinates',
                data=sorted_coords,
                compression='gzip',
                compression_opts=4
            )
            pt_group.create_dataset(
                'Masses',
                data=sorted_masses,
                compression='gzip',
                compression_opts=4
            )

            # Write cell structure for this rank
            write_cell_structure(f_out, cell_counts, cell_offsets, args.cdim, cell_size)

            # Copy header to first rank's file, or synthesize if BoxSize provided
            if rank == 0:
                if args.copy_header and args.header_key in f_in:
                    f_in.copy(args.header_key, f_out, 'Header')
                elif args.boxsize is not None:
                    write_minimal_header(f_out, boxsize, total_particles)

        print(f"Rank {rank}: Wrote {count} particles to {rank_file}")

    comm.Barrier()

    # Rank 0 creates virtual dataset
    if rank == 0:
        print(f"Rank 0: Creating virtual file {virtual_file}")
        create_virtual_file(
            base_name, size, total_particles, args.particle_type,
            args.copy_header or args.boxsize is not None, args.cdim, boxsize
        )
        print(f"✓ Conversion complete: {virtual_file}")


def create_virtual_file(base_name, nranks, total_particles, particle_type,
                        include_header, cdim, boxsize):
    """
    Create a virtual HDF5 file that combines all rank files.

    Args:
        base_name: Base output filename (without .hdf5)
        nranks: Number of MPI ranks
        total_particles: Total number of particles across all ranks
        particle_type: Particle type group name
        include_header: Whether to include header from rank 0
        cdim: Number of cells per dimension
        boxsize: Box dimensions [X, Y, Z]
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

    # Get particle counts from each rank file and merge cell structures
    particle_counts = []
    ncells = cdim ** 3
    merged_cell_counts = np.zeros(ncells, dtype=np.int32)

    for i in range(nranks):
        rank_file = f"{base_name}_rank_{i}.hdf5"
        with h5py.File(rank_file, 'r') as f:
            npart = f[f'{particle_type}/Masses'].shape[0]
            particle_counts.append(npart)

            # Merge cell counts from each rank
            if 'Cells/Counts/PartType1' in f:
                merged_cell_counts += f['Cells/Counts/PartType1'][:]

    # Recompute global offsets from merged counts
    merged_cell_offsets = np.zeros(ncells, dtype=np.int64)
    merged_cell_offsets[1:] = np.cumsum(merged_cell_counts[:-1])

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

        # Write merged cell structure
        cell_size = boxsize / cdim
        write_cell_structure(f, merged_cell_counts, merged_cell_offsets, cdim, cell_size)

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
