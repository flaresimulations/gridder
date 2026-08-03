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


def write_grid_points(parts, dataset_path, output_file, chunk_size, overwrite):
    """Write all centres atomically in gridder text format."""
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if output_file.exists() and not overwrite:
        raise ValueError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write("# SWIFT FOF halo centres for the gridder\n")
            output.write(f"# HDF5 dataset: {dataset_path}\n")

            for part in parts:
                with h5py.File(part, "r") as handle:
                    dataset = handle[dataset_path]
                    for start in range(0, dataset.shape[0], chunk_size):
                        centres = np.asarray(
                            dataset[start : start + chunk_size], dtype=np.float64
                        )
                        if not np.all(np.isfinite(centres)):
                            raise ValueError(
                                f"Non-finite centre coordinates found in {part}, "
                                f"rows {start}:{start + centres.shape[0]}"
                            )
                        np.savetxt(output, centres, fmt="%.17g")

        temporary_path.replace(output_file)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    try:
        parts = discover_parts(args.fof_directory, args.expected_parts)
        dataset_path = choose_dataset(parts[0], args.centres_dataset)
        data_parts, counts = inspect_parts(parts, dataset_path)
        write_grid_points(
            data_parts,
            dataset_path,
            args.output_file,
            args.chunk_size,
            args.overwrite,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Read dataset: {dataset_path}")
    print(
        f"Found {len(parts)} catalogue parts; read centre data from "
        f"{len(data_parts)} part(s) containing {sum(counts)} halos"
    )
    print(f"Wrote grid points to: {args.output_file}")


if __name__ == "__main__":
    main()
