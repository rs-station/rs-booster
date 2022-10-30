import reciprocalspaceship as rs
import numpy as np


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