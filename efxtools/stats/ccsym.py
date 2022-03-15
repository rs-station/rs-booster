#!/user/bin/env python
"""
Compute CCsym from careless output.

Note: This method currently assumes that careless has been called using
both the reduced-symmetry and the parent spacegroup
"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "mtz",
        nargs="+",
        help="MTZs containing crossvalidation data from careless",
    )
    parser.add_argument(
        "--op",
        required=True,
        help=(
            "Symmetry operation to use to compute internal difference map. "
            "Can be given as ISYM if used with a `spacegroup` argument"
        ),
    )

    # Optional arguments
    parser.add_argument(
        "-sg",
        "--spacegroup",
        help=(
            "Spacegroup to use for symmetry operation "
            "(only necessary if `op` specifies an ISYM)."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help=("Method for computing correlation coefficient (spearman or pearson)"),
    )
    parser.add_argument(
        "--mod2",
        action="store_true",
        help=("Use id mod 2 to assign delays (use when employing spacegroup hack)"),
    )

    return parser.parse_args()


def make_halves_ccsym(mtz, op, bins=10):
    """Construct half-datasets for computing CCsym"""
    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    temp1 = half1.merge(
        half1.apply_symop(op).hkl_to_asu(),
        on=["H", "K", "L", "repeat"],
        suffixes=("1", "2"),
    )
    temp2 = half2.merge(
        half2.apply_symop(op).hkl_to_asu(),
        on=["H", "K", "L", "repeat"],
        suffixes=("1", "2"),
    )

    temp1["DF"] = temp1["F1"] - temp1["F2"]
    temp2["DF"] = temp2["F1"] - temp2["F2"]

    temp = temp1[["DF", "repeat"]].merge(
        temp2[["DF", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def analyze_ccsym_mtz(
    mtzpath, op, bins=10, return_labels=True, method="spearman", mod2=False
):
    """Compute CCsym from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)
    m, labels = make_halves_ccsym(mtz, op)

    grouper = m.groupby(["bin", "repeat"])[["DF1", "DF2"]]
    result = (
        grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
    )

    if mod2:
        result["delay"] = np.floor(int(mtzpath[-5]) / 2)
    else:
        result["delay"] = int(mtzpath[-5])

    if return_labels:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()

    # Get symmetry operation
    try:
        isym = int(args.op)
        sg = gemmi.SpaceGroup(args.spacegroup)
        op = sg.operations().sym_ops[isym]
    except ValueError:
        op = gemmi.Op(args.op)

    results = []
    labels = None
    for m in args.mtz:
        result = analyze_ccsym_mtz(m, op, method=args.method, mod2=args.mod2)
        if result is None:
            continue
        else:
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CCsym"] = results[("DF1", "DF2")]
    results.drop(columns=[("DF1", "DF2")], inplace=True)

    print(results)

    sns.lineplot(
        data=results, x="bin", y="CCsym", hue="delay", ci="sd", palette="viridis"
    )
    plt.xticks(range(10), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$CC_{sym}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
