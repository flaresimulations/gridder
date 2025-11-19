"""A script to generate grid point files from SOAP catalogues for FOF halos.

Note that this script uses the InputHalos field to extract the FOF centres
and masses from the SOAP catalogues.
"""

import argparse

import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grid point position files from the FOF "
        "(InputHalos) in SOAP catalogues."
    )
    parser.add_argument(
        "soap_catalogue",
        type=str,
        help="Path to the SOAP catalogue HDF5 file.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="fof_grid_points.txt",
        type=str,
        help="Output file for the grid point positions.",
    )
    return parser.parse_args()


def main():
    """Main function to generate grid point files from SOAP catalogues.

    This will produce a whitespace delimited text file containing the centres
    and a second (single column) file containing the masses of the FOF halos
    from the InputHalos field in the SOAP catalogue.

    Note that all mass = 0 halos are excluded from the output.
    """
    args = parse_args()

    # Open the SOAP catalogue
    with h5py.File(args.soap_catalogue, "r") as soap_file:
        input_halos = soap_file["InputHalos/FOF"]
        positions = input_halos["Centres"][...]
        masses = input_halos["Masses"][...]

    # Exclude halos with mass = 0
    valid_indices = masses > 0
    positions = positions[valid_indices]
    masses = masses[valid_indices]

    # Create the strings we will write to the text files
    position_lines = [f"{pos[0]} {pos[1]} {pos[2]}\n" for pos in positions]
    mass_lines = [f"{mass}\n" for mass in masses]

    # Write the positions to the output file
    with open(args.output_file, "w") as pos_file:
        pos_file.writelines(position_lines)

    # Write the masses to a separate file
    mass_output_file = args.output_file.replace(".h5", "_masses.txt")
    with open(mass_output_file, "w") as mass_file:
        mass_file.writelines(mass_lines)


if __name__ == "__main__":
    main()
