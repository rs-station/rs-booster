#!/usr/bin/env python
"""
Make an ordinary difference map between columns of data.

The naming convention chosen for inputs is `on` and `off`, such
that the generated differrence map will be `|F_on| - |F_off|`.
"""

import argparse
import numpy as np
import reciprocalspaceship as rs

from rsbooster.diffmaps.weights import compute_weights
from rsbooster.utils.io import subset_to_FSigF


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-on",
        "--onmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=("MTZ to be used as `on` data. Specified as (filename, F, SigF)"),
    )
    parser.add_argument(
        "-off",
        "--offmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=("MTZ to be used as `off` data. Specified as (filename, F, SigF)"),
    )
    parser.add_argument(
        "-r",
        "--refmtz",
        nargs=2,
        metavar=("ref", "phi_col"),
        required=True,
        help=(
            "MTZ containing isomorphous phases to be used. "
            "Specified as (filename, Phi)."
        ),
    )

    # Optional arguments
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help="alpha value for computing difference map weights (default=0.0)",
    )
    parser.add_argument(
        "-d",
        "--dmax",
        type=float,
        default=None,
        help="If set, dmax to truncate difference map",
    )
    parser.add_argument(
        "-o", "--outfile", default="diffmap.mtz", help="Output MTZ filename"
    )

    return parser.parse_args()


def main():

    # Parse commandline arguments
    args = parse_arguments()
    refmtz, phi_col = args.refmtz

    # Read MTZ files
    onmtz = subset_to_FSigF(*args.onmtz, {args.onmtz[1]: "F", args.onmtz[2]: "SigF"})
    offmtz = subset_to_FSigF(
        *args.offmtz, {args.offmtz[1]: "F", args.offmtz[2]: "SigF"}
    )

    ref = rs.read_mtz(refmtz)
    ref.rename(columns={phi_col: "Phi"}, inplace=True)
    ref = ref.loc[:, ["Phi"]]
    if not isinstance(ref["Phi"].dtype, rs.PhaseDtype):
        raise ValueError(
            f"{args.Phi} is not a phases column in {args.mtz2}. Try again."
        )

    diff = onmtz.merge(offmtz, on=["H", "K", "L"], suffixes=("_on", "_off"))
    diff["DF"] = diff["F_on"] - diff["F_off"]
    diff["SigDF"] = np.sqrt((diff["SigF_on"] ** 2) + (diff["SigF_off"] ** 2))

    # Compute weights
    diff["W"] = compute_weights(diff["DF"], diff["SigDF"], alpha=args.alpha)
    diff["W"] = diff["W"].astype("Weight")

    # Join with phases and write map
    common = diff.index.intersection(ref.index).sort_values()
    diff = diff.loc[common]
    diff["Phi"] = ref.loc[common, "Phi"]
    diff.infer_mtz_dtypes(inplace=True)

    if args.dmax is None:
        diff.write_mtz(args.outfile)
    else:
        diff = diff.loc[diff.compute_dHKL()["dHKL"] < args.dmax]
        diff.write_mtz(args.outfile)


if __name__ == "__main__":
    main()
