#!/user/bin/env python
"""
Compute Rsplit from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import reciprocalspaceship as rs
from rsbooster.stats import summary_stats
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
        default="by_bin",
        choices=["by_bin", "global"],
        help=("Method for computing Rsplit, either with a linear scale per bin or a global linear scale"),
    )
    parser.add_argument(
        "-b",
        "--bins",
        default=10,
        type=int,
        help=("Number of bins for scaling (default: 10)")
    )
    parser.add_argument(
        "-f",
        "--amplitudes",
        dest='amplitudes', 
        action='store_true', 
        default=False,
        help=("Calculate an Rsplit(F). By default (False), amplitudes are converted to intensities before Rsplit is calculated.")
    )

    return parser.parse_args()


def calculate_rsplit(I1, I2, k):
    return np.sqrt(2)*np.sum(np.abs(I1 - k*I2)) / np.sum(np.abs(I1 + k*I2))


def estimate_rsplit_scaling_coefficient(I1, I2):
    def loss(k):
        return calculate_rsplit(I1, I2, k)

    p = minimize(loss, x0=1.)
    return p.x


def rsplit_from_dataset(ds, key_1, key_2, by_bin=True):
    """
    Calculate Rsplit for (key_1, key_2), with a linear scale per bin (if by_bin) or otherwise globally.
    Report Rsplit by bin (if overall=False) or overall.
    """
    bins=ds.bin.unique()
    
    ds["k"]=1.
    
    if "repeat" not in ds.columns:
        ds["repeat"]=0
    repeats=ds.repeat.unique()
    
    # there is probably a more pandas-esque way to do this with .apply() or so.
    for repeat in repeats:
        if by_bin:
            for bin in bins:
                k = estimate_rsplit_scaling_coefficient(
                    ds.loc[(ds.bin==bin) & (ds.repeat==repeat), key_1].to_numpy(),
                    ds.loc[(ds.bin==bin) & (ds.repeat==repeat), key_2].to_numpy(),
                )
                # print(f"k: {k}")
                ds.loc[(ds.bin==bin) & (ds.repeat==repeat), "k"]=k[0]
        else:
            k = estimate_rsplit_scaling_coefficient(
                ds.loc[ds.repeat==repeat, key_1].to_numpy(),
                ds.loc[ds.repeat==repeat, key_2].to_numpy(),
            )
            ds.loc[ds.repeat==repeat, "k"]=k[0]
        

    result={}
    for repeat in repeats:
        rsplit_list=[]
        for bin in bins:
            rsplit = calculate_rsplit(
                ds.loc[(ds.bin==bin) & (ds.repeat==repeat), key_1].to_numpy(),
                ds.loc[(ds.bin==bin) & (ds.repeat==repeat), key_2].to_numpy(),
                ds.loc[(ds.bin==bin) & (ds.repeat==repeat),   "k"].to_numpy(),
            )
            rsplit_list.append(rsplit)
        result[repeat]=rsplit_list
    # Let's append the overall Rsplit too
    for repeat in repeats:
        rsplit = calculate_rsplit(
            ds.loc[ds.repeat==repeat, key_1].to_numpy(),
            ds.loc[ds.repeat==repeat, key_2].to_numpy(),
            ds.loc[ds.repeat==repeat,   "k"].to_numpy(),
        )
        result[repeat].append(rsplit)
    return result


def make_halves_rsplit(mtz, bins=10,generate_I=True):
    """Construct half-datasets for computing Rsplit
    
    Keyword arguments:
    mtz -- rs dataset containing a "half" column specifying the origin of each merged reflection
    bins -- integer specifying the number of resolution bins to assign
    generate_I -- boolean, specifies whether to generate merged intensity estimates from F and sigF
    """
      
    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    temp = half1[["F", "SigF", "repeat"]].merge(
        half2[["F", "SigF", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    if generate_I and "I1" not in temp.columns and "I2" not in temp.columns:
        temp["I1"]=temp["F1"]**2 + temp["SigF1"]**2 
        temp["I2"]=temp["F2"]**2 + temp["SigF2"]**2 
    temp.dropna(inplace=True)

    return temp, labels


def analyze_rsplit_mtz(mtzpath, bins=10, generate_I=True, return_labels=True, by_bin=True):
    """Compute Rsplit from 2-fold cross-validation
       
       mtzpath -- string specifying the path to the MTZ to be analyzed OR an rs.DataSet object.
       bins -- number of resolution bins (integer, default: 10)
       type -- string specifying intensities (I) or structure factor amplitudes (F) (default: I)
       return_labels -- boolean indicating whether to return resolution bin labels (default: True)
       by_bin -- boolean specifying whether to use a linear scale per bin (default: True) or globally.

       As written, this function assumes Careless output with F and SigF columns present.
    """
    
    if type(mtzpath) is rs.dataset.DataSet:
        mtz=mtzpath
    else:
        mtz = rs.read_mtz(mtzpath)
    
    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation or generate an\
        MTZ that contains merged F or I for half-datasets with a 'half' column distinguishing their \
        half-dataset origin (e.g., 0/1)")

    m, labels = make_halves_rsplit(mtz, bins, generate_I)
    labels = labels + ["Overall"]
    
    if generate_I:
        key_1="I1"
        key_2="I2"
    else:
        key_1="F1"
        key_2="F2"

    result = rsplit_from_dataset(m, key_1=key_1, key_2=key_2, by_bin=by_bin)
    # print(result)
    
    result = rs.DataSet(data=result)#.transpose()
    result = result.stack()
    result = result.reset_index()
    result = result.rename(columns={"level_0":"bin", "level_1":"repeat",0: "Rsplit"})
    # print(result)
    
    if return_labels:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()
    nbins=args.bins
    # print(f"We are scaling by {args.method} (determined by -m flag)") #by_bin
    # print(f"Using {args.bins} resolution bins (determined by -b flag)") #None
    # print(f"We could calculate an Rsplit(F), for amplitudes. Are we? {args.amplitudes}.") #False

    if args.method=="by_bin":
        by_bin=True
    else:
        by_bin=False
    
    results = []
    plt.figure(figsize=(6,4))
    for m in args.mtz:
        result,labels = analyze_rsplit_mtz(m, bins=nbins, generate_I=(not args.amplitudes), return_labels=True, by_bin=by_bin)
        if result is None:
            continue
        else:
            print("\n\nAnalyzing " + m)
            print("Rsplit for each repeat & averaged over repeats; across resolution bins and overall:\n")
            r_split_all= summary_stats.parse_xval_stats(result, labels, nbins, name="Rsplit")        
            print(r_split_all.head(args.bins+1))

            result["filename"]=m
            # print(result.info())
            results.append(result)    
    
    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["bin"]=results["bin"].astype(int)
    results["Rsplit"]=results["Rsplit"].astype(float)
    # results.columns = [x[0] for x in results.columns] # get rid of pesky tuples

            
    sns.lineplot(
        data=results.loc[results.bin < nbins,], x="bin", y="Rsplit", errorbar="sd", hue="filename",palette="viridis"
    )
    plt.xticks(range(nbins), labels[:nbins], rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$R_{split}$" + f"({args.method})")
    plt.ylim([0, 1])
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
            

if __name__ == "__main__":
    main()
