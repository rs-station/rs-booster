#!/user/bin/env python
"""
Compute CCanom from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns


from rsbooster.stats.parser import BaseParser
class ArgumentParser(BaseParser):
    def __init__(self):
        super().__init__(
            description=__doc__
        )
        
        # Required arguments
        self.add_argument(
            "mtz",
            nargs="+",
            help="MTZs containing crossvalidation data from careless",
        )
    
        self.add_argument(
            "-m",
            "--method",
            default="spearman",
            choices=["spearman", "pearson"],
            help=("Method for computing correlation coefficient (spearman or pearson)"),
        )


def make_halves_ccanom(mtz, bins=10):
    """Construct half-datasets for computing CCanom"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    half1["DF"] = half1["F(+)"] - half1["F(-)"]
    half2["DF"] = half2["F(+)"] - half2["F(-)"]

    temp = half1[["DF", "repeat"]].merge(
        half2[["DF", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def analyze_ccanom_mtz(mtzpath, bins=10, return_labels=True, method="spearman"):
    """Compute CCsym from 2-fold cross-validation"""

    if type(mtzpath) is rs.dataset.DataSet:
        mtz=mtzpath
    else:
        mtz = rs.read_mtz(mtzpath)
        
    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation")

    if "F(+)" not in mtz.columns:
        raise ValueError("Please provide MTZs merged with `--anomalous` in careless")

    mtz = mtz.acentrics
    mtz = mtz.loc[(mtz["N(+)"] > 0) & (mtz["N(-)"] > 0)]
    m, labels = make_halves_ccanom(mtz)

    grouper = m.groupby(["bin", "repeat"])[["DF1", "DF2"]]
    result = (
        grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
    )

    if return_labels:
        return result, labels
    else:
        return result


def run_analysis(args):
    results = []
    labels = None
    for m in args.mtz:
        result = analyze_ccanom_mtz(m, method=args.method)
        if result is None:
            continue
        else:
            result[0]["filename"] = m
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CCanom"] = results[("DF1", "DF2")]
    results.drop(columns=[("DF1", "DF2")], inplace=True)

    for k in ('bin', 'repeat'):
        results[k] = results[k].to_numpy('int32')

    if args.output is not None:
        results.to_csv(args.output)
    else:
        print(results.to_string())

    # print(results.info())

    sns.lineplot(
        data=results, x="bin", y="CCanom", hue="filename", errorbar="sd", palette="viridis"
    )
    plt.xticks(range(10), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$CC_{anom}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()

def parse_arguments():
    return ArgumentParser()

def main():
    run_analysis(parse_arguments().parse_args())
