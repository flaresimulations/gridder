"""Extract gridder input positions from a split SWIFT FOF output."""

import argparse
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np


COMMON_CENTRE_DATASETS = (
    "Groups/Centres",
    "Groups/Centers",
    "Groups/CentreOfPotential",
    "Groups/CenterOfPotential",
    "FOF/Centres",
    "FOF/Centers",
    "Group/Centres",
    "Group/Centers",
)

COMMON_MASS_DATASETS = (
    "Groups/Masses",
    "Groups/Mass",
    "FOF/Masses",
    "FOF/Mass",
    "Group/Masses",
    "Group/Mass",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract halo centres from a split SWIFT FOF HDF5 output and write "
            "a whitespace-delimited gridder input file."
        )
    )
    parser.add_argument(
        "fof_directory",
        type=Path,
        help=(
            "SWIFT FOF output directory containing files such as "
            "fof_output_0077/fof_output_0077.0.hdf5."
        ),
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("fof_grid_points.txt"),
        help="Output coordinate file (default: fof_grid_points.txt).",
    )
    parser.add_argument(
        "--centres-dataset",
        type=str,
        help=(
            "HDF5 path to the halo centres. If omitted, common paths are "
            "checked and an unambiguous centre-like (N, 3) dataset is used."
        ),
    )
    parser.add_argument(
        "--mass-output-file",
        "-m",
        type=Path,
        default=Path("fof_masses.txt"),
        help="Output halo mass file (default: fof_masses.txt).",
    )
    parser.add_argument(
        "--masses-dataset",
        type=str,
        help=(
            "HDF5 path to the FOF halo masses. If omitted, common paths are "
            "checked and an unambiguous mass-like dataset is used."
        ),
    )
    parser.add_argument(
        "--expected-parts",
        type=int,
        default=32,
        help="Expected number of SWIFT FOF output parts (default: 32).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of centres read and written at once (default: 100000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def discover_parts(directory, expected_parts):
    """Return split catalogue files in numerical part order."""
    if not directory.is_dir():
        raise ValueError(f"FOF directory does not exist: {directory}")

    prefix = directory.name
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.hdf5$")
    parts = {}

    for path in directory.glob(f"{prefix}.*.hdf5"):
        match = pattern.match(path.name)
        if match:
            part = int(match.group(1))
            if part in parts:
                raise ValueError(f"Duplicate FOF part {part}: {path}")
            parts[part] = path

    if not parts:
        raise ValueError(
            f"No files matching {prefix}.N.hdf5 were found in {directory}"
        )

    if expected_parts <= 0:
        raise ValueError("--expected-parts must be positive")

    expected = set(range(expected_parts))
    missing = sorted(expected - set(parts))
    unexpected = sorted(set(parts) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing parts: {missing}")
        if unexpected:
            details.append(f"unexpected parts: {unexpected}")
        raise ValueError(
            f"Expected parts 0-{expected_parts - 1}; " + "; ".join(details)
        )

    return [parts[index] for index in range(expected_parts)]


def centre_like_datasets(handle):
    """Find numeric (N, 3) datasets with centre-like names."""
    matches = []

    def inspect(name, item):
        final_name = name.rsplit("/", maxsplit=1)[-1].lower()
        if (
            isinstance(item, h5py.Dataset)
            and len(item.shape) == 2
            and item.shape[1] == 3
            and np.issubdtype(item.dtype, np.number)
            and ("centre" in final_name or "center" in final_name)
        ):
            matches.append(name)

    handle.visititems(inspect)
    return matches


def mass_like_datasets(handle):
    """Find numeric one-dimensional datasets with mass-like names."""
    matches = []

    def inspect(name, item):
        final_name = name.rsplit("/", maxsplit=1)[-1].lower()
        if (
            isinstance(item, h5py.Dataset)
            and (len(item.shape) == 1 or (len(item.shape) == 2 and item.shape[1] == 1))
            and np.issubdtype(item.dtype, np.number)
            and (final_name == "mass" or final_name == "masses")
        ):
            matches.append(name)

    handle.visititems(inspect)
    return matches


def choose_dataset(first_part, requested_dataset):
    """Resolve the centre dataset path from the first catalogue part."""
    with h5py.File(first_part, "r") as handle:
        if requested_dataset:
            dataset = requested_dataset.strip("/")
            if dataset not in handle:
                raise ValueError(
                    f"Dataset '{dataset}' does not exist in {first_part}"
                )
            return dataset

        for dataset in COMMON_CENTRE_DATASETS:
            if dataset in handle:
                return dataset

        candidates = centre_like_datasets(handle)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"Could not find a centre-like (N, 3) dataset in {first_part}. "
                "Specify its HDF5 path with --centres-dataset."
            )
        raise ValueError(
            "Multiple possible centre datasets were found in "
            f"{first_part}: {candidates}. Select one with --centres-dataset."
        )


def choose_mass_dataset(first_part, requested_dataset):
    """Resolve the FOF mass dataset path from the first catalogue part."""
    with h5py.File(first_part, "r") as handle:
        if requested_dataset:
            dataset = requested_dataset.strip("/")
            if dataset not in handle:
                raise ValueError(
                    f"Dataset '{dataset}' does not exist in {first_part}"
                )
            return dataset

        for dataset in COMMON_MASS_DATASETS:
            if dataset in handle:
                return dataset

        candidates = mass_like_datasets(handle)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"Could not find a mass-like dataset in {first_part}. Specify "
                "its HDF5 path with --masses-dataset."
            )
        raise ValueError(
            f"Multiple possible mass datasets were found in {first_part}: "
            f"{candidates}. Select one with --masses-dataset."
        )


def inspect_parts(parts, dataset_path):
    """Find and validate centre arrays stored in all parts or only part zero."""
    availability = []
    for part in parts:
        with h5py.File(part, "r") as handle:
            availability.append(dataset_path in handle)

    if all(availability):
        data_parts = parts
    elif availability[0] and not any(availability[1:]):
        data_parts = parts[:1]
    else:
        present = [index for index, available in enumerate(availability) if available]
        missing = [index for index, available in enumerate(availability) if not available]
        raise ValueError(
            f"Dataset '{dataset_path}' has an ambiguous split layout; "
            f"present in parts {present} and absent from parts {missing}"
        )

    counts = []
    for part in data_parts:
        with h5py.File(part, "r") as handle:
            dataset = handle[dataset_path]
            if (
                not isinstance(dataset, h5py.Dataset)
                or len(dataset.shape) != 2
                or dataset.shape[1] != 3
                or not np.issubdtype(dataset.dtype, np.number)
            ):
                raise ValueError(
                    f"Dataset '{dataset_path}' in {part} must be a numeric "
                    f"(N, 3) array; found shape={dataset.shape}, dtype={dataset.dtype}"
                )
            counts.append(dataset.shape[0])
    return data_parts, counts


def inspect_mass_parts(parts, dataset_path, expected_counts):
    """Validate that one mass is stored for every extracted centre."""
    if len(parts) != len(expected_counts):
        raise ValueError(
            f"Internal part/count mismatch: {len(parts)} parts and "
            f"{len(expected_counts)} expected counts"
        )
    counts = []
    for part, expected_count in zip(parts, expected_counts):
        with h5py.File(part, "r") as handle:
            if dataset_path not in handle:
                raise ValueError(f"Dataset '{dataset_path}' is absent from {part}")
            dataset = handle[dataset_path]
            valid_shape = len(dataset.shape) == 1 or (
                len(dataset.shape) == 2 and dataset.shape[1] == 1
            )
            if (
                not isinstance(dataset, h5py.Dataset)
                or not valid_shape
                or not np.issubdtype(dataset.dtype, np.number)
            ):
                raise ValueError(
                    f"Dataset '{dataset_path}' in {part} must be a numeric "
                    f"one-dimensional array; found shape={dataset.shape}, "
                    f"dtype={dataset.dtype}"
                )
            if dataset.shape[0] != expected_count:
                raise ValueError(
                    f"Centre/mass count mismatch in {part}: {expected_count} "
                    f"centres but {dataset.shape[0]} masses"
                )
            counts.append(dataset.shape[0])
    return counts


def write_catalogue(
    parts,
    centres_dataset,
    masses_dataset,
    output_file,
    mass_output_file,
    chunk_size,
    overwrite,
):
    """Write aligned centre and mass text files atomically."""
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    for path in (output_file, mass_output_file):
        if path.exists() and not overwrite:
            raise ValueError(
                f"Output file already exists: {path}. Use --overwrite to replace it."
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    mass_output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as output, tempfile.NamedTemporaryFile(
            mode="w",
            dir=mass_output_file.parent,
            prefix=f".{mass_output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as mass_output:
            temporary_paths = [Path(output.name), Path(mass_output.name)]
            output.write("# SWIFT FOF halo centres for the gridder\n")
            output.write(f"# HDF5 dataset: {centres_dataset}\n")
            mass_output.write("# SWIFT FOF halo masses in catalogue units\n")
            mass_output.write(f"# HDF5 dataset: {masses_dataset}\n")

            for part in parts:
                with h5py.File(part, "r") as handle:
                    positions = handle[centres_dataset]
                    masses = handle[masses_dataset]
                    for start in range(0, positions.shape[0], chunk_size):
                        centres = np.asarray(
                            positions[start : start + chunk_size], dtype=np.float64
                        )
                        mass_values = np.asarray(
                            masses[start : start + chunk_size], dtype=np.float64
                        ).reshape(-1)
                        if not np.all(np.isfinite(centres)):
                            raise ValueError(
                                f"Non-finite centre coordinates found in {part}, "
                                f"rows {start}:{start + centres.shape[0]}"
                            )
                        if not np.all(np.isfinite(mass_values)):
                            raise ValueError(
                                f"Non-finite halo masses found in {part}, rows "
                                f"{start}:{start + mass_values.shape[0]}"
                            )
                        np.savetxt(output, centres, fmt="%.17g")
                        np.savetxt(mass_output, mass_values, fmt="%.17g")

        temporary_paths[0].replace(output_file)
        temporary_paths[1].replace(mass_output_file)
    except Exception:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    try:
        parts = discover_parts(args.fof_directory, args.expected_parts)
        centres_dataset = choose_dataset(parts[0], args.centres_dataset)
        data_parts, counts = inspect_parts(parts, centres_dataset)
        masses_dataset = choose_mass_dataset(data_parts[0], args.masses_dataset)
        inspect_mass_parts(data_parts, masses_dataset, counts)
        write_catalogue(
            data_parts,
            centres_dataset,
            masses_dataset,
            args.output_file,
            args.mass_output_file,
            args.chunk_size,
            args.overwrite,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Read centres: {centres_dataset}")
    print(f"Read masses: {masses_dataset}")
    print(
        f"Found {len(parts)} catalogue parts; read centre data from "
        f"{len(data_parts)} part(s) containing {sum(counts)} halos"
    )
    print(f"Wrote grid points to: {args.output_file}")
    print(f"Wrote aligned masses to: {args.mass_output_file}")


if __name__ == "__main__":
    main()
