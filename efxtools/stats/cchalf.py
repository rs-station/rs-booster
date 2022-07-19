#!/user/bin/env python
"""
Compute CChalf from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
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
        "-m",
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help=("Method for computing correlation coefficient (spearman or pearson)"),
    )

    return parser.parse_args()


def make_halves_cchalf(mtz, bins=10):
    """Construct half-datasets for computing CChalf"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    temp = half1[["F", "repeat"]].merge(
        half2[["F", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def analyze_cchalf_mtz(mtzpath, bins=10, return_labels=True, method="spearman"):
    """Compute CChalf from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation")

    m, labels = make_halves_cchalf(mtz)

    grouper = m.groupby(["bin", "repeat"])[["F1", "F2"]]
    result = (
        grouper.corr(method=method).unstack()[("F1", "F2")].to_frame().reset_index()
    )

    if return_labels:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()

    results = []
    labels = None
    for m in args.mtz:
        result = analyze_cchalf_mtz(m, method=args.method)
        if result is None:
            continue
        else:
            result[0]["filename"] = m
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CChalf"] = results[("F1", "F2")]
    results.drop(columns=[("F1", "F2")], inplace=True)

    print(results)

    sns.lineplot(
        data=results, x="bin", y="CChalf", hue="filename", ci="sd", palette="viridis"
    )
    plt.xticks(range(10), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$CC_{1/2}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
