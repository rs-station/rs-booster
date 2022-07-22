#!/usr/bin/env python
"""
Make an internal difference map using the given symmetry operation
"""

import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

from rsbooster.diffmaps.weights import compute_weights
from rsbooster.utils.io import subset_to_FSigF


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--inputmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=(
            "MTZ to be used for internal difference map. "
            "Specified as (filename, F, SigF)"
        ),
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
    parser.add_argument(
        "-op",
        "--symop",
        required=True,
        help=(
            "Symmetry operation to use to compute internal difference map. "
            "Can be given as ISYM if used with a `spacegroup` argument."
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
        "-sg",
        "--spacegroup",
        help="Spacegroup to use for symmetry operation (only necessary if `op` specifies an ISYM).",
    )
    parser.add_argument(
        "-o", "--outfile", default="internal_diffmap.mtz", help="Output MTZ filename"
    )

    return parser.parse_args()


def main():

    # Parse commandline arguments
    args = parse_arguments()
    refmtz, phi_col = args.refmtz

    # Read MTZ files
    mtz = subset_to_FSigF(
        *args.inputmtz, {args.inputmtz[1]: "F", args.inputmtz[2]: "SigF"}
    )
    ref = rs.read_mtz(refmtz)

    # Canonicalize column names
    ref.rename(columns={phi_col: "Phi"}, inplace=True)
    ref = ref[["Phi"]]

    # Error checking of datatypes
    if not isinstance(ref["Phi"].dtype, rs.PhaseDtype):
        raise ValueError(
            f"{args.Phi} is not a phases column in {args.mtz2}. Try again."
        )

    # Compare across symmetry operation
    try:
        isym = int(args.symop)
        sg = gemmi.SpaceGroup(args.spacegroup)
        op = sg.operations().sym_ops[isym]
    except ValueError:
        op = gemmi.Operation(args.symop)

    internal = mtz.merge(
        mtz.apply_symop(op).hkl_to_asu(), on=["H", "K", "L"], suffixes=("1", "2")
    )
    internal["DF"] = internal["F1"] - internal["F2"]
    internal["SigDF"] = np.sqrt((internal["SigF1"] ** 2) + (internal["SigF2"] ** 2))

    # Compute weights
    internal["W"] = compute_weights(internal["DF"], internal["SigDF"], alpha=args.alpha)
    internal["W"] = internal["W"].astype("Weight")

    # Join with phases and write map
    common = internal.index.intersection(ref.index).sort_values()
    internal = internal.loc[common]
    internal["Phi"] = ref.loc[common, "Phi"]
    internal.infer_mtz_dtypes(inplace=True)

    if args.dmax is None:
        internal.write_mtz(args.outfile)
    else:
        internal = internal.loc[internal.compute_dHKL()["dHKL"] < args.dmax]
        internal.write_mtz(args.outfile)


if __name__ == "__main__":
    main()
