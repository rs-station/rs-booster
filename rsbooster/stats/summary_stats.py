import reciprocalspaceship as rs
import pandas as pd
import numpy as np


def parse_xval_stats(xval_stats, labels, nbins, name="CChalf"):
    if name=="CChalf":
        xval_stats[name] = xval_stats[("F1", "F2")].astype('float')
        xval_stats.drop(columns=[("F1", "F2")], inplace=True)
    if name=="CCanom":
        xval_stats[name] = xval_stats[("DF1", "DF2")].astype('float')
        xval_stats.drop(columns=[("DF1", "DF2")], inplace=True)

    xval_stats_repeat_avg=xval_stats.groupby(by=["bin"]).mean().drop(columns=["repeat"]).rename({name:name+" (avg)"})

    xval_stats_by_repeat={}
    for repeat in xval_stats.repeat.unique():
        xval_stats_by_repeat[repeat]=xval_stats.loc[xval_stats.repeat==repeat,]
        xval_stats_by_repeat[repeat].set_index(keys=["bin"],inplace=True)
        xval_stats_by_repeat[repeat]=xval_stats_by_repeat[repeat].drop(columns=["repeat"])
    xval_stats_by_repeat["avg"]=xval_stats_repeat_avg
    xval_stats_by_repeat_all=rs.concat(xval_stats_by_repeat,check_isomorphous=False,axis=1)

    labels=pd.DataFrame(data={"Res. bin (A)": labels})
    labels.loc[nbins, "Res. bin (A)"]="Overall"
    xval_stats_all=pd.concat([labels, xval_stats_by_repeat_all.reset_index(drop=True)],axis=1)
    xval_stats_all.set_index(keys=["Res. bin (A)"],inplace=True)
    
    return xval_stats_all


def calculate_multiplicity(merged):
    """
    Calculate multiplicity for Careless output by bin (bins must exist already).
    The last row will contain overall multiplicity.
    """
    
    if "F(+)" in merged.columns:
        merged=merged.stack_anomalous()
    
    nbins=len(merged.bin.unique())
    
    multiplicity = merged.groupby("bin")["N"].mean()
    multiplicity.name = "multiplicity"
    multiplicity.loc[nbins] = merged["N"].mean()
    
    return multiplicity


def calculate_FsigF(merged):
    """
    Compute average F/SigF for Careless output by bin (bins must exist already).
    Note that this function, at present, adds a f/sigf column to its input.
    The last row will contain overall F/sigF.
    """ 
    if "F(+)" in merged.columns:
        merged=merged.stack_anomalous()

    nbins=len(merged.bin.unique())
    
    merged["F/SigF"] = merged["F"] / merged["SigF"]
    
    f_sigf = merged.groupby("bin")["F/SigF"].mean()
    f_sigf.name = "<F/SigF>merged"
    f_sigf.loc[nbins] = merged["F/SigF"].mean()
    
    return f_sigf


def cc_star_from_cchalf(cchalf):
    
    ccstar = np.sqrt((2 * cchalf) / (1 + cchalf))
    if isinstance(ccstar, rs.DataSet):
        ccstar.name = "CCstar"
    
    return ccstar