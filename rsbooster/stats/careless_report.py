import argparse
import glob
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from rsbooster.stats import cchalf, ccanom, ccpred, rsplit, summary_stats 


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-p",
        "--prefix",
        dest='prefix',
        default="thermolysin_",
        required=True,
        help=("Prefix for the Careless output in this directory (without trailing underscore)"),
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest='dir',
        default="./",
        help=("Directory/ or path/ pointing to location with careless output"),
    )
    parser.add_argument(
        "-b",
        "--bins",
        default=10,
        type=int,
        help=("Number of bins for reporting (default: 10)")
    )
    parser.add_argument(
        "-m",
        "--method",
        default="spearman",
        help=("Method for calculating correlation coefficients (spearman or pearson)")
    )
    parser.add_argument(
        "-dmin",
        "--dmin", 
        dest='dmin', 
        default=1.0,
        type=float,
        help=("High-resolution cutoff for reporting.") 
    )
    parser.add_argument(
        "-dmax",
        "--dmax", 
        dest='dmax', 
        default=100.0,
        type=float,
        help=("Low-resolution cutoff for reporting.") 
    )

    return parser.parse_args()


def main():
    # Parse commandline arguments
    args = parse_arguments()
    nbins=args.bins
    custom_range = sorted([args.dmin,args.dmax], key=float)
    workdir = args.dir
    
    pd.set_option("display.precision", 5)
    pd.set_option('display.max_columns', None)
    
    mtz_list     =glob.glob(args.dir + args.prefix + '_?.mtz')
    mtz_xval_list=glob.glob(args.dir + args.prefix + '_xval_?.mtz')
    mtz_pred_list=glob.glob(args.dir + args.prefix + '_predictions_?.mtz')
    
    for m in mtz_list:       
        ds=rs.read_mtz(m)
        ds.compute_dHKL(inplace=True) 
        ds=ds.loc[(ds.dHKL>=custom_range[0]) & \
                  (ds.dHKL< custom_range[1]),ds.columns]
        ds, labels = ds.assign_resolution_bins(nbins) 
        
        labels=pd.DataFrame(data={"res. (A)": labels})
        labels.loc[nbins, "res. (A)"]="Overall"
        
        print( "\nReading " + m)
        print(f"The dataset contains {len(ds):,} rows.")
        if "F(+)" in ds.columns:
            anomalous=True
            print("Treating these as as anomalous data")
        else:
            anomalous=False
            print("No anomalous flags detected. Treating as non-anomalous data")
        
        # COMPLETENESS
        completeness=rs.stats.completeness.compute_completeness(ds,bins=nbins,anomalous=anomalous)
        completeness = pd.concat([labels, completeness.reset_index(drop=True)],axis=1)

        print("\nCompleteness: \n")
        print(completeness.head(nbins+1))
        
        # MULTIPLICITY & F/sigF
        multiplicity = summary_stats.calculate_multiplicity(ds)
        FsigF        = summary_stats.calculate_FsigF(ds)
        results=pd.concat([labels, multiplicity,FsigF],axis=1)
        print("\n\nMultiplicity & F/sigF:\n")
        print(results.head(nbins+1))

        
    for m in mtz_xval_list:
        print("\n\nReporting crossvalidation statistics for " + m +":")

        ds_xval=rs.read_mtz(m)
        ds_xval.compute_dHKL(inplace=True)
        ds_xval, labels = ds_xval.assign_resolution_bins(nbins) 
        ds_xval=ds_xval.loc[(ds_xval.dHKL>=custom_range[0]) &\
                            (ds_xval.dHKL< custom_range[1]),ds_xval.columns]
        
        # CC1/2
        cc_half, _ =cchalf.analyze_cchalf_mtz(ds_xval, bins=nbins, method=args.method)
        cc_half_all=summary_stats.parse_xval_stats(cc_half, labels, nbins, name="CChalf")
        
        print("\n\nCC1/2 for each repeat & averaged over repeats; across resolution bins and overall:\n")
        print(cc_half_all.head(nbins+1))
        
        
        # Rsplit
        r_split, _ = rsplit.analyze_rsplit_mtz(ds_xval, bins=nbins, by_bin=True)
        r_split_all= summary_stats.parse_xval_stats(r_split, labels, nbins, name="Rsplit")
        
        print("\n\nRsplit for each repeat & averaged over repeats; across resolution bins and overall:\n")
        print(r_split_all.head(nbins+1))
        
        # CCanom
        if anomalous:
            cc_anom, _ = ccanom.analyze_ccanom_mtz(ds_xval, bins=nbins, method=args.method)
            cc_anom_all=summary_stats.parse_xval_stats(cc_anom, labels, nbins, name="CCanom")
        
            print("\n\nCCanom for each repeat & averaged over repeats; across resolution bins and overall:\n")
            print(cc_anom_all.head(nbins+1))

    for m in mtz_pred_list:
        print("\n\nReporting CCpred statistics for " + m +":")
        ds_pred=rs.read_mtz(m)
        ds_pred.compute_dHKL(inplace=True)
        ds_pred, labels = ds_pred.assign_resolution_bins(nbins) 
        ds_pred=ds_pred.loc[(ds_pred.dHKL>=custom_range[0]) & \
                            (ds_pred.dHKL< custom_range[1]),ds_pred.columns]

        cc_pred, labels = ccpred.compute_ccpred(ds_pred)
        cc_pred["CCpred"] = cc_pred[("Iobs", "Ipred")]
        cc_pred.drop(columns=[("Iobs", "Ipred")], inplace=True)
        

        cc_pred_by_test={}
        for test in cc_pred.test.unique():
            cc_pred_by_test[test]=cc_pred.loc[cc_pred.test==test,]
            cc_pred_by_test[test].set_index(keys=["bin"],inplace=True)
            cc_pred_by_test[test]=cc_pred_by_test[test].drop(columns=["test"])
        cc_pred_by_test_all=rs.concat(cc_pred_by_test,check_isomorphous=False,axis=1)

        labels=pd.DataFrame(data={"Res. bin (A)": labels})
        labels.loc[nbins, "Res. bin (A)"]="Overall"
        cc_pred_all=pd.concat([labels, cc_pred_by_test_all.reset_index(drop=True)],axis=1)
        cc_pred_all.set_index(keys=["Res. bin (A)"],inplace=True)

        print("\n\nCCpred (train=0, test=1) across resolution bins and overall:\n")
        print(cc_pred_all.head(nbins+1))

if __name__ == "__main__":
    main()
