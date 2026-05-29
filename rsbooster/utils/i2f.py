import argparse
import reciprocalspaceship as rs
from reciprocalspaceship.algorithms import scale_merged_intensities
from pathlib import Path
import structlog

OUTPUT_MTZ_DEFAULT_NAME: str = "FW_scaled.mtz"
INFER_COLUMN_NAME: str = "infer"

log = structlog.get_logger()


# TODO: the logic below should be more robust - but that should be done upstream in `rs`
def infer_intensity_column(ds: rs.DataSet) -> str:
    """Find the first column with IntensityDtype in the dataset."""
    for col in ds.columns:
        if isinstance(ds[col].dtype, rs.IntensityDtype):
            return col
    raise ValueError(
        f"Could not infer intensity column. No columns with IntensityDtype found. "
        f"Available columns: {list(ds.columns)}"
    )


def infer_uncertainty_column(ds: rs.DataSet, intensity_column: str) -> str:
    """look for a column with the same name as `intensity_column` with a `SIG` prefix"""
    candidate_column_name = f"SIG{intensity_column}"
    if candidate_column_name in ds.columns:
        if isinstance(ds[candidate_column_name].dtype, rs.StandardDeviationDtype):
            return candidate_column_name

    raise ValueError(
        f"Could not infer uncertainty column for intensity column '{intensity_column}'. "
        f"looking for: {candidate_column_name}"
        f"Available columns: {list(ds.columns)}"
    )


def parse_arguments(
    command_line_arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate structure factor amplitudes (`F`) based on observed intensities (`I`)"
        " and uncertainties (`SIG`) using the French-Wilson model."
    )
    parser.add_argument(
        "input_mtz",
        help="Input MTZ file (with merged intensities & corresponding uncertainties)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_MTZ_DEFAULT_NAME,
        help="Output MTZ file name (default: `FW_scaled.mtz`)",
    )
    parser.add_argument(
        "--inplace",
        default=False,
        action="store_true",
        help="add the computed columns to the input MTZ, rather than generating a new MTZ file",
    )
    parser.add_argument(
        "-i",
        "--intensity-column",
        default=INFER_COLUMN_NAME,
        help="Column name for intensity (default: infer from input)",
    )
    parser.add_argument(
        "-u",
        "--uncertainties-column",
        default=INFER_COLUMN_NAME,
        help="Column name for intensity uncertainties (sigmas) (default: infer from input)",
    )
    parser.add_argument(
        "-p",
        "--output-prefix",
        default="FW-",
        help="The MTZ column prefix to append to the French-Wilson output. For example, the "
        "default value is `FW`, which will generate new columns with the names "
        "`FW-I`, `FW-SIGI`, `FW-F`, `FW-SIGF`. To request no prefix, pass `-p ''`",
    )

    parser.add_argument(
        "--method",
        choices=["isotropic", "anisotropic"],
        default="anisotropic",
        help="Method to estimate mean intensity (default: anisotropic)",
    )
    parser.add_argument(
        "--number-of-bins",
        type=int,
        default=40,
        help="Number of resolution bins (used with isotropic method)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=2.0,
        help="Bandwidth (used with anisotropic method)",
    )
    parser.add_argument(
        "--minimum-sigma",
        type=float,
        default=None,
        help="Minimum allowed value of the local scale parameter `sigma`.",
    )

    args = parser.parse_args(command_line_arguments)

    if not Path(args.input_mtz).exists():
        msg = f"cannot find input file: {args.input_mtz}"
        raise IOError(msg)

    if args.inplace and args.output != OUTPUT_MTZ_DEFAULT_NAME:
        msg = "inplace modification of the input MTZ and an output MTZ filename were both "
        msg += "specified -- suspect this is an error, please pick one!"
        raise ValueError(msg)

    return args


def main(command_line_arguments: list[str] | None = None) -> None:
    args = parse_arguments(command_line_arguments)

    ds = rs.read_mtz(args.input_mtz)

    # Infer column names if not specified
    intensity_column = args.intensity_column
    if intensity_column == INFER_COLUMN_NAME:
        intensity_column = infer_intensity_column(ds)
        log.info("Inferred intensity column", column=intensity_column)

    uncertainties_column = args.uncertainties_column
    if uncertainties_column == INFER_COLUMN_NAME:
        uncertainties_column = infer_uncertainty_column(ds, intensity_column)
        log.info("Inferred uncertainties column", column=uncertainties_column)

    # Validate intensity column type
    if not isinstance(ds[intensity_column].dtype, rs.IntensityDtype):
        raise ValueError(
            f"Column '{intensity_column}' is not an intensity type. "
            f"Expected IntensityDtype, got {ds[intensity_column].dtype}"
        )

    output_columns = [
        f"{args.output_prefix}{column}" for column in ["I", "SIGI", "F", "SIGF"]
    ]

    french_wilson_ds = scale_merged_intensities(
        ds,
        intensity_key=intensity_column,
        sigma_key=uncertainties_column,
        output_columns=output_columns,
        inplace=False,
        mean_intensity_method=args.method,
        bins=args.number_of_bins,
        bw=args.bandwidth,
        minimum_sigma=args.minimum_sigma,
    )

    if args.inplace:
        french_wilson_ds.write_mtz(args.input_mtz)
        log.info(
            "Wrote French-Wilson structure factor amplitudes", output=args.input_mtz
        )
    else:
        french_wilson_ds.write_mtz(args.output)
        log.info("Wrote French-Wilson structure factor amplitudes", output=args.output)


if __name__ == "__main__":
    main()
