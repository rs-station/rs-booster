#!/user/bin/env python
"""
Compute CCanom from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
from rsbooster.stats import summary_stats
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
    parser.add_argument(
        "-b",
        "--bins",
        default=10,
        type=int,
        help=("Number of bins for scaling (default: 10)")
    )   

    return parser.parse_args()


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
    m, labels = make_halves_ccanom(mtz, bins=bins)

    grouper = m.groupby(["bin", "repeat"])[["DF1", "DF2"]]
    result = (
        grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
    )
    # DH addition: 
    grouper = m.groupby(["repeat"])[["DF1", "DF2"]]
    result_overall = (
        grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
    )
    result_overall["bin"]=bins
    result = rs.concat([result, result_overall],check_isomorphous=False,ignore_index=True)
    labels = labels + ["Overall"]

    if return_labels:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()
    nbins = args.bins

    results = []
    labels = None
    for m in args.mtz:
        result, labels = analyze_ccanom_mtz(m, bins=args.bins, method=args.method)
        if result is None:
            continue
        else:
            print("\n\nAnalyzing " + m)
            print("CCanom for each repeat & averaged over repeats; across resolution bins and overall:\n")
            cc_anom_all= summary_stats.parse_xval_stats(result, labels, nbins, name="CCanom")        
            print(cc_anom_all.head(nbins+1))
            
            result["filename"]=m
            # print(result)
            results.append(result)
    
    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["bin"]=results["bin"].astype(int)
    results["CCanom"]=results["CCanom"].astype(float)
    results.columns = [x[0] for x in results.columns] # get rid of pesky tuples

    sns.lineplot(
        data=results.loc[results.bin < nbins,], x="bin", y="CCanom", errorbar="sd", hue="filename",palette="viridis"
    )
    plt.xticks(range(nbins), labels[:nbins], rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$CC_{anom}$ " + f"({args.method})")
    plt.legend() #loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.ylim([-0.1,1])
    plt.show()

            


if __name__ == "__main__":
    main()
