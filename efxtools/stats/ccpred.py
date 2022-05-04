#!/user/bin/env python
"""
Compute CCsym from careless output.

Note: If more than one `-i` argument is given, only overall CCpred is reported
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
        "-i",
        "--inputmtzs",
        nargs="+",
        action="append",
        required=True,
        help="MTZs containing holdout prediction data from careless",
    )

    # Optional arguments
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
        help=("Use (id mod 2) to assign delays (use when employing spacegroup hack)"),
    )

    return parser.parse_args()


def compute_ccpred(
    mtzpath, overall=False, bins=10, return_labels=True, method="spearman", mod2=False
):
    """Compute CCsym from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    if overall:
        grouper = mtz.groupby(["test"])[["Iobs", "Ipred"]]
    else:
        mtz, labels = mtz.assign_resolution_bins(bins)
        grouper = mtz.groupby(["bin", "test"])[["Iobs", "Ipred"]]

    result = (
        grouper.corr(method=method)
        .unstack()[("Iobs", "Ipred")]
        .to_frame()
        .reset_index()
    )

    result["id"] = mtzpath.split("/")[0]
    if mod2:
        result["delay"] = np.floor(int(mtzpath[-5]) / 2)
    else:
        result["delay"] = int(mtzpath[-5])
    result["spacegroup"] = mtz.spacegroup.xhm()

    if return_labels and not overall:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()

    results = []
    labels = None

    if isinstance(args.inputmtzs[0], list) and len(args.inputmtzs) > 1:
        overall = True
        mtzs = [item for sublist in args.inputmtzs for item in sublist]
    else:
        overall = False
        mtzs = [item for sublist in args.inputmtzs for item in sublist]

    for m in mtzs:
        result = compute_ccpred(m, overall=overall, method=args.method, mod2=args.mod2)
        if isinstance(result, tuple):
            results.append(result[0])
            labels = result[1]
        else:
            results.append(result)

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CCpred"] = results[("Iobs", "Ipred")]
    results.drop(columns=[("Iobs", "Ipred")], inplace=True)

    print(results)
    if overall:
        g = sns.relplot(
            data=results,
            x="id",
            y="CCpred",
            style="test",
            hue="delay",
            col="spacegroup",
            kind="line",
            palette="viridis",
        )
        for col_val, ax in g.axes_dict.items():
            ax.grid(True)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
        plt.show()
    else:
        g = sns.relplot(
            data=results,
            x="bin",
            y="CCpred",
            style="test",
            hue="delay",
            col="spacegroup",
            kind="line",
            palette="viridis",
        )
        for col_val, ax in g.axes_dict.items():
            ax.set_xticks(range(10))
            ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
            ax.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
