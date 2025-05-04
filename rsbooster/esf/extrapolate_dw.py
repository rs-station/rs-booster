#!/usr/bin/env python
"""
Bayesian extrapolated structure factors calculation using the double-wilson statistical model.

Equations
---------
The underlying model assumes that

- Prior distribution: ground-state (GS) and excited-state (ES) amplitudes are correlated with strenght r (see https://pmc.ncbi.nlm.nih.gov/articles/PMC11291090/)
- the ON structure factor amplitudes are a "random-diffuse" mixture of GS and ES (a la Phil Coppens)
  with 
  E^ON = p E^ES + (1-p) E^GS

Notes
-----
The algorithm currently assumes that the input MTZs contain structure factor amplitudes scaled and merged using Careless.
It expects to find the following: {high, low, loc, scale} parametrizing a truncated normal.

"""

import argparse
import numpy   as np
from tqdm      import tqdm
from scipy.stats import truncnorm, rice, foldnorm
import reciprocalspaceship as rs


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-on",
        "--onmtz",
        nargs='+',
        required=True,
        help="MTZ to be used as `on` data.",
    )
    parser.add_argument(
        "-off",
        "--offmtz",
        nargs='+',
        required=True,
        help=("MTZ to be used as `off` data."),
    )
    parser.add_argument(
        "-path",
        "--mtzpath",
        default="./",
        help=("MTZ to be used as `off` data."),
    )

    parser.add_argument(
        "-n", "--nsamples", type=int, default=200_000, help="number of samples for pseudo-integration"
    )
    parser.add_argument(
        "-r", "--rDW", type=float, default=0.9, help="Assumed DW correlation between GS and ES structure factors"
    )
    parser.add_argument(
        "-p", "--es-fraction", type=float, help="Excited state fraction"
    )
    parser.add_argument(
        "-f", "--factor", type=float, help="Extrapolation factor"
    )
    parser.add_argument(
        "-o", "--outfile", default="esf_dw.mtz", help="Output MTZ filename"
    )
    parser.add_argument( #not working?
        "--disable-progress-bar", action='store_true', help="Disable the progress bar"
    )
    return parser


# THIS HAS BEEN LIFTED FROM https://github.com/rs-station/reciprocalspaceship/blob/main/reciprocalspaceship/algorithms/scale_merged_intensities.py
# IT WOULD BE BETTER TO IMPORT DIRECTLY INSTEAD!
# OTHER ROUTINES (E.G. local scaling) MAY WORK BETTER
def mean_intensity_by_resolution(I, dHKL, bins=50, gridpoints=None):
    """
    Use a gaussian kernel smoother to compute mean intensities as a function of resolution.
    The kernel smoother is evaulated over the specified number of gridpoints and then interpolated.
    Kernel bandwidth is derived from `bins` as follows
    >>> X = dHKL**-2
    bw = (X.max() - X.min)/bins

    Parameters
    ----------
    I : array
        Array of observed intensities
    dHKL : array
        Array of reflection resolutions
    bins : float(optional)
        "bins" is used to determine the kernel bandwidth.
    gridpoints : int(optional)
        Number of gridpoints at which to estimate the mean intensity. This will default to 20*bins

    Returns
    -------
    Sigma : array
        Array of point estimates for the mean intensity at resolution in dHKL.
    """
    # Use double precision
    I = np.array(I, dtype=np.float64)
    dHKL = np.array(dHKL, dtype=np.float64)

    if gridpoints is None:
        gridpoints = int(bins * 20)

    X = dHKL**-2.0
    bw = (X.max() - X.min()) / bins

    # Evaulate the kernel smoother over grid points
    grid = np.linspace(X.min(), X.max(), gridpoints)
    K = np.exp(-0.5 * ((X[:, None] - grid[None, :]) / bw) ** 2.0)
    K = K / K.sum(0)
    protos = I @ K

    # Use a kernel smoother to interpolate the grid points
    bw = grid[1] - grid[0]
    K = np.exp(-0.5 * ((X[:, None] - grid[None, :]) / bw) ** 2.0)
    K = K / K.sum(1)[:, None]
    Sigma = K @ protos

    return Sigma

def w_avg(E, w, bReal = True):
    if bReal:
        E_avg  = np.real(np.sum(w*E))
    else: 
        E_avg  = np.sum(w*E)
    return E_avg

def abs_w_avg(E, w):
    E_avg  = np.real(np.sum(w*E))
    E_abs_avg = np.abs(E_avg)
    return E_abs_avg

def w_avg_abs(E, w):
    E_avg_abs = np.sum(w*np.abs(E))
    return E_avg_abs

def var_abs_E(E, w, E_ref=None):
    if E_ref is not None:
        _out = np.sum(w*np.abs(E - E_ref)**2)
    else:
        _out = np.sum(w*np.abs(E - w_avg(E,w,True))**2)
    return _out
    
def main():

    # Parse commandline arguments
    args = parse_arguments().parse_args()
    on_list  = args.onmtz
    off_list = args.offmtz
    r   = args.rDW
    print(args.outfile)
    print(f"Disable progress bar? {args.disable_progress_bar}")
    bReturnGS=False

    if args.factor and args.es_fraction:
        raise ValueError("Only specify `-f` or `-p`, not both.")
    elif args.factor:
        p = 1.0/args.factor
    else:
        p=args.es_fraction

    # Read MTZ files
    ds_on_list=[]
    for m in on_list:
        ds_on_list.append(rs.read_mtz(m))
    ds_on = ds_on_list[0]  # for now, we'll just deal with the first one. want to be able to provide multiple!
    
    ds_off_list=[]
    for m in off_list:
        ds_off_list.append(rs.read_mtz(m))
    ds_of = ds_off_list[0]  # for now, we'll just deal with the first one. want to be able to provide multiple!


    # Samples from the prior
    mean    = [0,0,0,0]                 
    cov     = 0.5*np.asarray([[1, 0, r, 0 ],\
                              [0, 1, 0, r ],\
                              [r, 0, 1, 0 ],\
                              [0, r, 0, 1 ]])
    E_1x_1y_2x_2y = np.random.multivariate_normal(mean=mean, cov=cov,size=args.nsamples)
        
    GS_ac = E_1x_1y_2x_2y[:,0] +1j*E_1x_1y_2x_2y[:,1]
    ES_ac = E_1x_1y_2x_2y[:,2] +1j*E_1x_1y_2x_2y[:,3]
    ph_GS = np.angle(GS_ac)
    GS_ac = GS_ac*np.exp(-1j*ph_GS)
    ES_ac = ES_ac*np.exp(-1j*ph_GS)
    
    OF_ac = GS_ac
    ON_ac = (1-p)*GS_ac + p*ES_ac
    ph_GS = np.angle(GS_ac)
    ph_ES = np.angle(ES_ac)
    ph_ON = np.angle(ON_ac)
    OF_ac_abs = np.abs( OF_ac)
    ON_ac_abs = np.abs( ON_ac)
    k = np.median(ON_ac_abs)/np.median(OF_ac_abs)
    ON_ac_abs = ON_ac_abs / k
    print(f"Scale factor to put acentric ON and OFF on same scale (k): {k:.5}")
  
    # CENTRIC CASE
    mean = [0,0]
    cov  = np.asarray([[1, r ],\
                       [r, 1 ]])
    E_1_2 = np.random.multivariate_normal(mean=mean, cov=cov,size=args.nsamples)
        
    GS_c = E_1_2[:,0]
    ES_c = E_1_2[:,1]
    ph_GS = np.angle(GS_c)
    GS_c = np.real(GS_c*np.exp(-1j*ph_GS)) # remove numerical contamination with imag comp
    ES_c = np.real(ES_c*np.exp(-1j*ph_GS))
        
    OF_c = GS_c
    ON_c = (1-p)*GS_c + p*ES_c
    OF_c_abs = np.abs( OF_c)
    ON_c_abs = np.abs( ON_c)
    k = np.median(ON_c_abs)/np.median(OF_c_abs)
    ON_c_abs = ON_c_abs / k
    print(f"Scale factor to put  centric ON and OFF on same scale (k): {k:.5}")
    
    ds_all = ds_of.merge(ds_on, left_index=True, right_index=True, suffixes=("_off", "_on"))
    ds_all.label_centrics(inplace=True)
    ds_all.compute_multiplicity(inplace=True)
    ds_all.compute_dHKL(inplace=True)    
    
    multiplicity = ds_all.EPSILON.to_numpy()
    
    Sigma_off=mean_intensity_by_resolution((ds_all.loc[:, "F_off"].to_numpy()**2)/multiplicity, ds_all.dHKL, bins=50, gridpoints=None)
    Sigma_on =mean_intensity_by_resolution((ds_all.loc[:, "F_on"].to_numpy()**2)/multiplicity, ds_all.dHKL, bins=50, gridpoints=None)

    row=ds_of.iloc[0]
    eps=1e-10
    a, b = (float(row["low"]) -  float(row["loc"])) / float(row["scale"]), \
           (float(row["high"]) - float(row["loc"])) / float(row["scale"])

    GS_abs_2_list   =[]
    ES_abs_2_list   =[]
    SIGGS_abs_2_list=[]
    SIGES_abs_2_list=[]
    for n in tqdm(range(len(ds_all.index)),disable=args.disable_progress_bar):
        row=ds_all.iloc[n]
    
        # set up the likelihood function
        rv_off = truncnorm(a, b, loc=float(row["loc_off"]), scale=float(row["scale_off"]))
        rv_on  = truncnorm(a, b, loc=float(row["loc_on"] ), scale=float(row["scale_on" ]))
        sqrt_Sig_off = np.sqrt(Sigma_off[n])
        sqrt_Sig_on  = np.sqrt(Sigma_on[ n])
        sqrt_eps = np.sqrt(float(row["EPSILON"]))
        if not row["CENTRIC"]:
            w_off=rv_off.pdf(sqrt_eps * sqrt_Sig_off * OF_ac_abs) 
            w_on =rv_on.pdf( sqrt_eps * sqrt_Sig_on  * ON_ac_abs)
            w = w_off * w_on * (w_off>eps)*(w_on>eps)
            sum_w = sum(w)
            if np.sum(w>0)>5:
                w = w/sum_w
                if bReturnGS:
                    GS_abs_2     = sqrt_eps * w_avg_abs(GS_ac,w)
                    SIGGS_abs_2  = sqrt_eps * np.sqrt( var_abs_E(GS_ac, w, 0) )
                ES_abs_2     = sqrt_eps * w_avg_abs(ES_ac,w)
                SIGES_abs_2  = sqrt_eps * np.sqrt( var_abs_E(ES_ac, w, 0) )
            else:
                if bReturnGS:
                    GS_abs_2     = np.nan
                    SIGGS_abs_2  = np.nan
                ES_abs_2     = np.nan
                SIGES_abs_2  = np.nan
        else: # row["CENTRIC"]:
            w_off=rv_off.pdf(sqrt_eps * sqrt_Sig_off * OF_c_abs)
            w_on =rv_on.pdf( sqrt_eps * sqrt_Sig_on * ON_c_abs)
            w = w_off * w_on * (w_off>eps)*(w_on>eps)
            sum_w = sum(w)
            if np.sum(w>0)>5:
                w = w/sum_w
                if bReturnGS:
                    GS_abs_2     = sqrt_eps * w_avg_abs(GS_c,w)
                    SIGES_abs_2  = sqrt_eps * np.sqrt( var_abs_E(GS_c, w, 0) )
                ES_abs_2     = sqrt_eps * w_avg_abs(ES_c,w)
                SIGES_abs_2  = sqrt_eps * np.sqrt( var_abs_E(ES_c, w, 0) )
            else:
                if bReturnGS:
                    GS_abs_2    = np.nan
                    SIGGS_abs_2 = np.nan
                ES_abs_2    = np.nan
                SIGES_abs_2 = np.nan
        if bReturnGS:
            GS_abs_2_list.append(GS_abs_2)
            SIGGS_abs_2_list.append(SIGGS_abs_2)
        ES_abs_2_list.append(ES_abs_2)
        SIGES_abs_2_list.append(SIGES_abs_2)

    # OUTPUT ES
    ds_all["ES_abs_2"]=ES_abs_2_list
    ds_all["SIGES_abs_2"]=SIGES_abs_2_list
    
    ds_all["ES_abs_2"]=ds_all["ES_abs_2"].astype("F")
    ds_all["SIGES_abs_2"]=ds_all["SIGES_abs_2"].astype("Q")
    
    ds_all.infer_mtz_dtypes(inplace=True)
    ds_all[["ES_abs_2","SIGES_abs_2","CENTRIC"]].write_mtz(args.outfile)

    if bReturnGS:
        # OUTPUT GS
        ds_all["GS_abs_2"]=GS_abs_2_list
        ds_all["SIGGS_abs_2"]=SIGGS_abs_2_list
        
        ds_all["GS_abs_2"]=ds_all["GS_abs_2"].astype("F")
        ds_all["SIGGS_abs_2"]=ds_all["SIGGS_abs_2"].astype("Q")
        
        ds_all.infer_mtz_dtypes(inplace=True)
        ds_all[["GS_abs_2","SIGGS_abs_2","CENTRIC"]].write_mtz(args.outfile[:-4]+"_GS_reference.mtz")


if __name__ == "__main__":
    main()
