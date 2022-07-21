#!/usr/bin/env python
"""
Run CCP4's scaleit on the given data.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import shutil
import subprocess
from tempfile import NamedTemporaryFile

import reciprocalspaceship as rs


def parse_arguments():
    """Parse commandline arguments"""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "-r",
        "--refmtz",
        nargs=3,
        metavar=("ref", "data_col", "sig_col"),
        required=True,
        help=(
            "MTZ to be used as reference for scaling using given data columns. "
            "Specified as (filename, F, SigF) or (filename, I, SigI)"
        ),
    )
    parser.add_argument(
        "-i",
        "--inputmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        action="append",
        required=True,
        help=(
            "MTZ to be scaled to reference using given data columns. "
            "Specified as (filename, F, SigF) or (filename, I, SigI)"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="scaled.mtz",
        help="MTZ file to which scaleit output will be written",
    )

    return parser.parse_args()


def load_mtz(mtzpath, data_col, sig_col):
    """Load mtz and do French-Wilson scaling, if necessary"""
    mtz = rs.read_mtz(mtzpath)

    # Check dtypes
    if not isinstance(
        mtz[data_col].dtype, (rs.StructureFactorAmplitudeDtype, rs.IntensityDtype)
    ):
        raise ValueError(
            f"{data_col} must specify an intensity or |F| column in {mtzpath}"
        )
    if not isinstance(mtz[sig_col].dtype, rs.StandardDeviationDtype):
        raise ValueError(
            f"{sig_col} must specify a standard deviation column in {mtzpath}"
        )

    # Run French-Wilson scaling if intensities are provided:
    if isinstance(mtz[data_col].dtype, rs.IntensityDtype):
        scaled = rs.algorithms.scale_merged_intensities(
            mtz, data_col, sig_col, mean_intensity_method="anisotropic"
        )
        result = scaled.loc[:, ["FW-F", "FW-SIGF"]]
        result.rename(columns={"FW-F": "F", "FW-SIGF": "SIGF"}, inplace=True)
        return result

    result = mtz.loc[:, [data_col, sig_col]]
    result.rename(columns={data_col: "F", sig_col: "SIGF"}, inplace=True)
    return result


def run_scaleit(joined, outfile, n_mtzs):
    """
    Run scaleit on given data

    Parameters
    ----------
    joined : filepath, str
        Path to MTZ file with input data
    outfile : filename, str
        Filename for scaled MTZ output
    n_mtzs : int
        Number of datasets being scaled to reference
    """

    columns = [f"FPH{i}=FPH{i} SIGFPH{i}=SIGFPH{i}" for i in range(1, n_mtzs + 1)]
    labin = " ".join(columns)
    with NamedTemporaryFile(suffix=".mtz") as tmp:
        joined.write_mtz(tmp.name)
        subprocess.call(
            f"scaleit HKLIN {tmp.name} HKLOUT {outfile} <<EOF\nrefine anisotropic\nLABIN FP=FP SIGFP=SIGFP {labin}\nEOF",
            shell=True,
        )

    return


def main():

    # Parse commandline arguments
    args = parse_arguments()

    # Test whether scaleit is on PATH
    if shutil.which("scaleit") is None:
        raise EnvironmentError(
            "Cannot find executable, scaleit. Please set up your CCP4 environment."
        )

    # Load reference
    ref = load_mtz(*args.refmtz)
    ref.rename(columns={"F": "FP", "SIGF": "SIGFP"}, inplace=True)

    # Load input datasets
    mtzs = []
    for i, inputmtz in enumerate(args.inputmtz, 1):
        mtz = load_mtz(*inputmtz)
        mtz.rename(columns={"F": f"FPH{i}", "SIGF": f"SIGFPH{i}"}, inplace=True)
        mtzs.append(mtz)

    # Join on common Miller indices
    common = None
    for mtz in mtzs:
        if common is None:
            common = ref.index.intersection(mtz.index)
        else:
            common = common.intersection(mtz.index)
    common = common.sort_values()

    print(f"Number of common reflections: {len(common)}")
    mtzs = [mtz.loc[common] for mtz in mtzs]
    joined = rs.concat([ref.loc[common]] + mtzs, axis=1)

    # Run scaleit
    run_scaleit(joined, args.outfile, len(mtzs))


if __name__ == "__main__":
    main()
