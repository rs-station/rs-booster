#!/usr/bin/env python
"""
Make an internal difference map using the given symmetry operation
"""

import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

from efxtools.diffmaps.weights import compute_weights


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument("mtz1", help="MTZ containing data")
    parser.add_argument("F", help="Column for |F| data in mtz1")
    parser.add_argument("SigF", help="Column for SigF data in mtz1")
    parser.add_argument("mtz2", help="MTZ containing phases")
    parser.add_argument("Phi", help="Column for phase data in mtz2")
    parser.add_argument(
        "op",
        help="Symmetry operation to use to compute internal difference map. Can be given as ISYM if used with a `spacegroup` argument",
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

    # Read MTZ files
    mtz = rs.read_mtz(args.mtz1)
    ref = rs.read_mtz(args.mtz2)

    # Canonicalize column names
    mtz.rename(columns={args.F: "F", args.SigF: "SigF"}, inplace=True)
    mtz = mtz[["F", "SigF"]]
    ref.rename(columns={args.Phi: "Phi"}, inplace=True)
    ref = ref[["Phi"]]

    # Error checking of datatypes
    if not isinstance(mtz["F"].dtype, rs.StructureFactorAmplitudeDtype):
        raise ValueError(
            f"{args.F} is not a structure factor amplitude in {args.mtz1}. Try again."
        )
    if not isinstance(mtz["SigF"].dtype, rs.StandardDeviationDtype):
        raise ValueError(
            f"{args.SigF} is not a structure factor amplitude in {args.mtz1}. Try again."
        )
    if not isinstance(ref["Phi"].dtype, rs.PhaseDtype):
        raise ValueError(
            f"{args.Phi} is not a phases column in {args.mtz2}. Try again."
        )

    # Compare across symmetry operation
    try:
        isym = int(args.op)
        sg = gemmi.SpaceGroup(args.spacegroup)
        op = sg.operations().sym_ops[isym]
    except ValueError:
        op = gemmi.Operation(args.op)

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
    internal.write_mtz(args.outfile)


if __name__ == "__main__":
    main()
