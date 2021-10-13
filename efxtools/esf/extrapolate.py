#!/usr/bin/env python
"""
Make extrapolated structure factors for refinement.

Equations
---------

    F_{esf}    = f * (F_{on} - F_{off}) + F_{calc}
    SigF_{esf} = sqrt( ( (f**2)*(SigF_{on}**2) ) + ( ((f**2)*(SigF_{off}**2) ) )

where f, is the extrapolation factor.

Notes
-----
    - F_{off} and F_{calc} can be the same MTZ file, as done in Hekstra et al, 
      Nature (2016). In that case, the equation for SigF_{esf} is adjusted to
      use (f-1)**2 for SigF_{off} to avoid double-counting in the error propagation.
"""

import argparse
import numpy as np

import reciprocalspaceship as rs

def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=__doc__)

    # Required arguments
    parser.add_argument("on_mtz", help="MTZ containing ON data")
    parser.add_argument("F_on", help="Column for F_on data in on_mtz")
    parser.add_argument("SigF_on", help="Column for SigF_on data in on_mtz")
    parser.add_argument("off_mtz", help="MTZ containing OFF data")
    parser.add_argument("F_off", help="Column for F_off data in off_mtz")
    parser.add_argument("SigF_off", help="Column for SigF_off data in off_mtz")
    parser.add_argument("calc_mtz", help="MTZ containing CALC data")
    parser.add_argument("F_calc", help="Column for F_calc data in calc_mtz")

    # Optional arguments
    parser.add_argument("-f", "--factor", type=float, default=10.0,
                        help="Extrapolation factor")
    parser.add_argument("-o", "--outfile", default="esf.mtz",
                        help="Output MTZ filename")

    return parser.parse_args()

def main():

    # Parse commandline arguments
    args = parse_arguments()

    # Read MTZ files
    on   = rs.read_mtz(args.on_mtz)
    off  = rs.read_mtz(args.off_mtz)
    calc = rs.read_mtz(args.calc_mtz)

    # Canonicalize column names
    on.rename(columns={args.F_on:"F", args.SigF_on:"SigF"}, inplace=True)
    off.rename(columns={args.F_off:"F", args.SigF_off:"SigF"}, inplace=True)
    calc.rename(columns={args.F_calc:"F_calc"}, inplace=True)

    # Subset DataSet objects to relevant columns
    on   = on[["F", "SigF"]]
    off  = off[["F", "SigF"]]
    calc = calc[["F_calc"]] 

    # Merge into common DataSet, keeping cell/spacegroup from on data
    joined = on.merge(off, on=["H", "K", "L"], suffixes=("_on", "_off"))
    joined = joined.merge(calc, on=["H", "K", "L"], suffixes=(None, "_calc"))

    # Compute F_esf and SigF_esf
    factor = args.factor
    joined["F_esf"] = factor*(joined["F_on"] - joined["F_off"]) + joined["F_calc"]
    if np.array_equal(joined["F_off"].to_numpy(), joined["F_calc"].to_numpy()):
        print("F_off == F_calc... changing error propagation accordingly.")
        joined["SigF_esf"] = np.sqrt(((factor**2)*(joined["SigF_on"]**2)) + (((factor-1)**2)*(joined["SigF_off"]**2)))
    else:
        joined["SigF_esf"] = np.sqrt(((factor**2)*(joined["SigF_on"]**2)) + ((factor**2)*(joined["SigF_off"]**2)))

    joined.infer_mtz_dtypes(inplace=True)
    joined.write_mtz(args.outfile)

if __name__ == "__main__":
    main()
