#!/usr/bin/env python
"""
Make extrapolated structure factors for refinement.

Equations
---------

with reference:

    F_{esf}    = f * (F_{on} - F_{off}) + F_{ref}
    SigF_{esf} = sqrt( ( (f**2)*(SigF_{on}**2) ) + ( (f**2)*(SigF_{off}**2) ) + (SigF_{ref}**2))

with calc:

    F_{esf}    = f * (F_{on} - F_{off}) + F_{calc}
    SigF_{esf} = sqrt( ( (f**2)*(SigF_{on}**2) ) + ( (f**2)*(SigF_{off}**2) ) )

where f, is the extrapolation factor.

Notes
-----
    - F_{off} and F_{calc} can be the same MTZ file, as done in Hekstra et al, 
      Nature (2016). In that case, the equation for SigF_{esf} is adjusted to
      use (f-1)**2 for SigF_{off} to avoid double-counting in the error propagation.
    - After computing |F_{esf}|, any negative structure factor amplitudes are converted
      to positive values. This is to ensure that they are handled correctly downstream in
      phenix, and because they are technically amplitudes of complex numbers and the phase
      should just be flipped by 180 degrees.
"""

import argparse
import numpy as np

import reciprocalspaceship as rs


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
        metavar=("mtz", "f_col", "sig_col"),
        required=True,
        help="MTZ to be used as `on` data. Specified as (filename, F, SigF)",
    )
    parser.add_argument(
        "-off",
        "--offmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=("MTZ to be used as `off` data. Specified as (filename, F, SigF)"),
    )

    # One of these must provided
    parser.add_argument(
        "-calc",
        "--calcmtz",
        nargs=2,
        metavar=("mtz", "data_col"),
        help=("MTZ to be used as `calc` data. Specified as (filename, F, SigF)"),
    )
    parser.add_argument(
        "-ref",
        "--refmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        help=("MTZ to be used as `ref` data. Specified as (filename, F, SigF)"),
    )

    # Optional arguments
    parser.add_argument(
        "-f", "--factor", type=float, default=10.0, help="Extrapolation factor"
    )
    parser.add_argument(
        "-o", "--outfile", default="esf.mtz", help="Output MTZ filename"
    )

    return parser.parse_args()


def main():

    # Parse commandline arguments
    args = parse_arguments()
    on, f_on, sigf_on = args.onmtz
    off, f_off, sigf_off = args.offmtz

    if args.calcmtz and args.refmtz:
        raise ValueError("Only specify `-calc` or `-ref`, not both.")
    elif args.calcmtz:
        calc, f_calc = args.calcmtz
        sigf_calc = None
    elif args.refmtz:
        calc, f_calc, sigf_calc = args.refmtz
    else:
        raise ValueError("One of `-calc` or `-ref` must be specified.")

    # Read MTZ files
    on = rs.read_mtz(on)
    off = rs.read_mtz(off)
    calc = rs.read_mtz(calc)

    # Canonicalize column names
    on.rename(columns={f_on: "F", sigf_on: "SigF"}, inplace=True)
    off.rename(columns={f_off: "F", sigf_off: "SigF"}, inplace=True)
    calc.rename(columns={f_calc: "F_calc"}, inplace=True)

    if sigf_calc:
        calc.rename(columns={sigf_calc: "SigF_calc"}, inplace=True)
        calc = calc[["F_calc", "SigF_calc"]]
    else:
        calc = calc[["F_calc"]]

    # Subset DataSet objects to relevant columns
    on = on[["F", "SigF"]]
    off = off[["F", "SigF"]]

    # Merge into common DataSet, keeping cell/spacegroup from on data
    joined = on.merge(off, on=["H", "K", "L"], suffixes=("_on", "_off"))
    joined = joined.merge(calc, on=["H", "K", "L"], suffixes=(None, "_calc"))

    # Compute F_esf and SigF_esf
    factor = args.factor
    joined["F_esf"] = factor * (joined["F_on"] - joined["F_off"]) + joined["F_calc"]
    if np.array_equal(joined["F_off"].to_numpy(), joined["F_calc"].to_numpy()):
        print("F_off == F_calc... changing error propagation accordingly.")
        joined["SigF_esf"] = np.sqrt(
            ((factor ** 2) * (joined["SigF_on"] ** 2))
            + (((factor - 1) ** 2) * (joined["SigF_off"] ** 2))
        )
    else:
        joined["SigF_esf"] = np.sqrt(
            ((factor ** 2) * (joined["SigF_on"] ** 2))
            + ((factor ** 2) * (joined["SigF_off"] ** 2))
        )

    if sigf_calc:
        joined["SigF_esf"] = np.sqrt(
            (joined["SigF_esf"] ** 2) + (joined["SigF_calc"] ** 2)
        )

    # Handle any negative values of |F_esf|
    joined["F_esf"] = np.abs(joined["F_esf"])

    joined.infer_mtz_dtypes(inplace=True)
    joined.write_mtz(args.outfile)


if __name__ == "__main__":
    main()
