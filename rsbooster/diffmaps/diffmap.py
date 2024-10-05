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

    return parser#.parse_args()

def remove_HKL_duplicates(ds, ds_name):
    num_duplicates = np.sum(ds.index.duplicated())
    if num_duplicates > 0:
        print(f"Warning: {ds_name} contains {num_duplicates} sets of duplicate Miller indices.")
        print( "Only the first instance of each HKL will be retained.")
        # useful diagnostic:
        # print(ds.reset_index().loc[ds.index.duplicated(keep=False),:].sort_values(["H","K","L"]).head(6))
        ds=ds.loc[~ds.index.duplicated(keep='first'),:]
    ds.merged=True
    return ds
    
def main():

    # Parse commandline arguments
    args = parse_arguments().parse_args()
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

    onmtz=onmtz.hkl_to_asu()
    offmtz=offmtz.hkl_to_asu()
    ref=ref.hkl_to_asu()

    onmtz = remove_HKL_duplicates(onmtz,  'ON MTZ')
    offmtz= remove_HKL_duplicates(offmtz, 'OFF MTZ')
    ref   = remove_HKL_duplicates(ref,    'REF MTZ')

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

    # Useful for PyMOL
    diff["wDF"] = (diff["DF"] * diff["W"]).astype("SFAmplitude")

    
    if args.dmax is None:
        diff.write_mtz(args.outfile)
    else:
        diff = diff.loc[diff.compute_dHKL()["dHKL"] < args.dmax]
        diff.write_mtz(args.outfile)


if __name__ == "__main__":
    main()
