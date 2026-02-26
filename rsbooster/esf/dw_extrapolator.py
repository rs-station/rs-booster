#!/usr/bin/env python
"""
Runs DW-Extrapolator, a Bayesian inference procedure to infer excited state structure factors in perturbative crystallography datsets.

Equations
---------
The underlying model assumes that ground state (GS) and excited state (ES) structure factors have correlation r and that the observed "on" state structure factors are given by F^{ON} = (1-p)*F^{GS} + p*F&{ES}. 

Notes
-----
    - At minimum, two .mtz's for the off and on data need to be provided
    - DW-Extrapolator can be run using French-Wilson scaled structure factors or integrated intensities
"""


import argparse
import numpy as np
from scipy.stats import truncnorm, norm
from scipy import optimize
import reciprocalspaceship as rs
import multiprocessing as mp
from multiprocessing import shared_memory
from reciprocalspaceship.algorithms.scale_merged_intensities import (
    mean_intensity_by_resolution,
)

try:                              
    from tqdm import tqdm         
except:                           
    tqdm = iter       

# globals
GS_ac_shm = ES_ac_shm = GS_c_shm = ES_c_shm = None
GS_ac = ES_ac = None
GS_c = ES_c = None


def init_shared_memory(GS_ac_name, ES_ac_name, GS_c_name, ES_c_name, nsamples):
    global GS_ac_shm, ES_ac_shm, GS_c_shm, ES_c_shm
    global GS_ac, ES_ac, GS_c, ES_c

    GS_ac_shm = shared_memory.SharedMemory(name=GS_ac_name)
    ES_ac_shm = shared_memory.SharedMemory(name=ES_ac_name)
    GS_c_shm = shared_memory.SharedMemory(name=GS_c_name)
    ES_c_shm = shared_memory.SharedMemory(name=ES_c_name)

    GS_ac = np.ndarray((nsamples,), dtype=np.complex64, buffer=GS_ac_shm.buf)
    ES_ac = np.ndarray((nsamples,), dtype=np.complex64, buffer=ES_ac_shm.buf)
    GS_c = np.ndarray((nsamples,), dtype=np.float32, buffer=GS_c_shm.buf)
    ES_c = np.ndarray((nsamples,), dtype=np.float32, buffer=ES_c_shm.buf)


# worker function for inference using the Truncated Normal distribution on structure factors
def estimate_reflection(args):
    (
        i,
        centric,
        loc_off,
        scale_off,
        loc_on,
        scale_on,
        sqrt_eps,
        sqrt_Sig_on,
        sqrt_Sig_off,
        p,
        eps,
        low,
        high,
        r,
    ) = args

    a_off, b_off = (float(low) - float(loc_off)) / float(scale_off), (
        float(high) - float(loc_off)
    ) / float(scale_off)

    a_on, b_on = (float(low) - float(loc_on)) / float(scale_on), (
        float(high) - float(loc_on)
    ) / float(scale_on)

    rv_off = truncnorm(a_off, b_off, loc=loc_off, scale=scale_off)
    rv_on = truncnorm(a_on, b_on, loc=loc_on, scale=scale_on)

    if not centric:
        ON = (1 - p) * GS_ac + p * ES_ac
        OF_abs = np.abs(GS_ac)
        ON_abs = np.abs(ON)
        k = np.median(ON_abs) / np.median(OF_abs)
        x_off = sqrt_eps * sqrt_Sig_off * OF_abs
        x_on = sqrt_eps * sqrt_Sig_on * (ON_abs / k)
        ES = ES_ac
    else:
        ON = (1 - p) * GS_c + p * ES_c
        OF_abs = np.abs(GS_c)
        ON_abs = np.abs(ON)
        k = np.median(ON_abs) / np.median(OF_abs)
        x_off = sqrt_eps * sqrt_Sig_off * OF_abs
        x_on = sqrt_eps * sqrt_Sig_on * (ON_abs / k)
        ES = ES_c

    # likelihood calculation
    logw = rv_off.logpdf(x_off) + rv_on.logpdf(x_on)
    ll_i = _logmeanexp(logw)

    # weights calculation
    m = np.max(logw)
    w = np.exp(logw - m)
    w *= w > eps
    sum_w = np.sum(w)

    if sum_w > 0 and np.sum(w > 0) > 5:
        w /= sum_w
        ES_abs = np.abs(ES)
        mean = np.sum(ES_abs * w)
        var = np.sum(w * (ES_abs) ** 2) - (mean) ** 2
        es_val = sqrt_eps * mean
        es_sig = sqrt_eps * np.sqrt(var)
        fs_val = sqrt_Sig_off * es_val
        fs_sig = sqrt_Sig_off * es_sig
        return (
            i,  # index
            es_val,  # ES_abs_2
            es_sig,  # SIGES_abs_2
            fs_val,  # FS_abs_2
            fs_sig,  # SIGFS_abs_2
            ll_i,
        )
    else:
        return (i, np.nan, np.nan, np.nan, np.nan, ll_i)


# worker function for inference using the Normal distribution for intensities
def estimate_reflection_intensity(args):
    (
        i,
        centric,
        I_off_obs,
        SigI_off,
        I_on_obs,
        SigI_on,
        Sigma_off,
        Sigma_on,
        p,
        eps,
        sqrt_eps,
        sqrt_Sig_on,
        sqrt_Sig_off,
        r,
    ) = args

    if not centric:
        ON = (1 - p) * GS_ac + p * ES_ac
        OF_abs = np.abs(GS_ac)
        ON_abs = np.abs(ON)
        k = np.median(ON_abs) / np.median(OF_abs)
        ON_abs = ON_abs / k

        loc_OF = OF_abs**2 * Sigma_off * sqrt_eps**2
        loc_ON = ON_abs**2 * Sigma_on * sqrt_eps**2
        ES = ES_ac
    else:
        ON = (1 - p) * GS_c + p * ES_c
        OF_abs = np.abs(GS_c)
        ON_abs = np.abs(ON)
        k = np.median(ON_abs) / np.median(OF_abs)
        ON_abs = ON_abs / k

        loc_OF = OF_abs**2 * Sigma_off * sqrt_eps**2
        loc_ON = ON_abs**2 * Sigma_on * sqrt_eps**2
        ES = ES_c

    # likelihood calculation
    logw = norm(loc=loc_OF, scale=SigI_off).logpdf(I_off_obs) + norm(
        loc=loc_ON, scale=SigI_on
    ).logpdf(I_on_obs)
    ll_i = _logmeanexp(logw)

    # weights calculation
    m = np.max(logw)
    w = np.exp(logw - m)
    w *= w > eps
    sum_w = np.sum(w)

    if sum_w > 0 and np.sum(w > 0) > 5:
        w /= sum_w
        ES_abs = np.abs(ES)
        mean = np.sum(ES_abs * w)
        var = np.sum(w * (ES_abs) ** 2) - (mean) ** 2
        es_val = sqrt_eps * mean
        es_sig = sqrt_eps * np.sqrt(var)
        fs_val = sqrt_Sig_off * es_val
        fs_sig = sqrt_Sig_off * es_sig
        return (
            i,  # index
            es_val,  # ES_abs_2
            es_sig,  # SIGES_abs_2
            fs_val,  # FS_abs_2
            fs_sig,  # SIGFS_abs_2
            ll_i,
        )
    else:
        return (i, np.nan, np.nan, np.nan, np.nan, ll_i)


def extrapolate_dw(args):
    """Run DW extrapolation given parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from :func:`parse_arguments`.

    Returns
    -------
    reciprocalspaceship.DataSet
        Output dataset containing extrapolated structure factor columns that were written
        to the output MTZ.
    """
    # Unpack arguments
    r = args.rDW
    nsamples = args.nsamples
    eps = 1e-10

    if args.factor and args.es_fraction:
        raise ValueError("Only specify `-f` or `-p`, not both.")
    elif args.factor:
        p = 1.0 / args.factor
    elif args.es_fraction is not None:
        p = args.es_fraction
    else:
        p = 0.125

    # Read MTZ files
    ds_on = rs.read_mtz(args.onmtz[0])
    ds_of = rs.read_mtz(args.offmtz[0])

    # Sample standard Multivariate Normals
    rng = np.random.default_rng(seed=args.seed)
    raw_Z_ac = rng.standard_normal((nsamples, 4)).astype(np.float32)  # acentric samples
    raw_Z_c = rng.standard_normal((nsamples, 2)).astype(np.float32)  # centric samples

    L_ac = np.sqrt(0.5) * np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [r, 0, np.sqrt(1 - r**2), 0],
            [0, r, 0, np.sqrt(1 - r**2)],
        ],
        dtype=np.float32,
    )

    E_ac = raw_Z_ac.dot(L_ac.T)
    GS_ac_local = (E_ac[:, 0] + 1j * E_ac[:, 1]).astype(np.complex64)
    ES_ac_local = (E_ac[:, 2] + 1j * E_ac[:, 3]).astype(np.complex64)

    L_c = np.array([[1, 0], [r, np.sqrt(1 - r**2)]], dtype=np.float32)
    E_c = raw_Z_c.dot(L_c.T)
    GS_c_local = E_c[:, 0].astype(np.float32)
    ES_c_local = E_c[:, 1].astype(np.float32)

    # Allocate storage

    GS_ac_shm_obj = shared_memory.SharedMemory(create=True, size=GS_ac_local.nbytes)
    ES_ac_shm_obj = shared_memory.SharedMemory(create=True, size=ES_ac_local.nbytes)
    GS_c_shm_obj = shared_memory.SharedMemory(create=True, size=GS_c_local.nbytes)
    ES_c_shm_obj = shared_memory.SharedMemory(create=True, size=ES_c_local.nbytes)

    np.ndarray(GS_ac_local.shape, dtype=GS_ac_local.dtype, buffer=GS_ac_shm_obj.buf)[
        :
    ] = GS_ac_local
    np.ndarray(ES_ac_local.shape, dtype=ES_ac_local.dtype, buffer=ES_ac_shm_obj.buf)[
        :
    ] = ES_ac_local
    np.ndarray(GS_c_local.shape, dtype=GS_c_local.dtype, buffer=GS_c_shm_obj.buf)[:] = (
        GS_c_local
    )
    np.ndarray(ES_c_local.shape, dtype=ES_c_local.dtype, buffer=ES_c_shm_obj.buf)[:] = (
        ES_c_local
    )

    # Merge and prepare data

    if args.use_intensities:  ##I/SigI case
        I_col, SigI_col = args.use_intensities
        ds_on = ds_on.rename(columns={I_col: "I", SigI_col: "SigI"})
        ds_on = ds_on.dropna(subset=["I", "SigI"], how="any")
        ds_of = ds_of.rename(columns={I_col: "I", SigI_col: "SigI"})
        ds_of = ds_of.dropna(subset=["I", "SigI"], how="any")

        ds_all = ds_of.merge(
            ds_on,
            left_index=True,
            right_index=True,
            suffixes=("_off", "_on"),
            check_isomorphous=False,
        )
        ds_all = ds_all.copy()
        ds_all.label_centrics(inplace=True)
        ds_all.compute_multiplicity(inplace=True)
        ds_all.compute_dHKL(inplace=True)
        multiplicity = ds_all.EPSILON.to_numpy()
        sqrt_eps_arr = np.sqrt(multiplicity)
    elif args.use_structure_factors:  ##Structure Factor case
        F_col, SigF_col = args.use_structure_factors
        ds_on = ds_on.rename(columns={F_col: "F", SigF_col: "SigF"})
        ds_on = ds_on.dropna(subset=["F", "SigF"], how="any")
        ds_of = ds_of.rename(columns={F_col: "F", SigF_col: "SigF"})
        ds_of = ds_of.dropna(subset=["F", "SigF"], how="any")

        ds_on = reparam(ds_on)
        ds_of = reparam(ds_of)
        ds_all = ds_of.merge(
            ds_on,
            left_index=True,
            right_index=True,
            suffixes=("_off", "_on"),
            check_isomorphous=False,
        )
        ds_all = ds_all.copy()
        ds_all.label_centrics(inplace=True)
        ds_all.compute_multiplicity(inplace=True)
        ds_all.compute_dHKL(inplace=True)
        multiplicity = ds_all.EPSILON.to_numpy()
        sqrt_eps_arr = np.sqrt(multiplicity)
    else:
        ds_on = ds_on.dropna(subset=["F", "SigF"], how="any")
        ds_of = ds_of.dropna(subset=["F", "SigF"], how="any")
        ds_all = ds_of.merge(
            ds_on,
            left_index=True,
            right_index=True,
            suffixes=("_off", "_on"),
            check_isomorphous=False,
        )
        ds_all = ds_all.copy()
        ds_all.label_centrics(inplace=True)
        ds_all.compute_multiplicity(inplace=True)
        ds_all.compute_dHKL(inplace=True)
        multiplicity = ds_all.EPSILON.to_numpy()
        sqrt_eps_arr = np.sqrt(multiplicity)

    # Prepare arguments for multiprocessing in working functions
    if not args.use_intensities:
        Sigma_off = mean_intensity_by_resolution(
            (ds_all.F_off**2 / multiplicity).to_numpy(), ds_all.dHKL.to_numpy()
        )
        Sigma_on = mean_intensity_by_resolution(
            (ds_all.F_on**2 / multiplicity).to_numpy(), ds_all.dHKL.to_numpy()
        )
        sqrt_Sigma_off = np.sqrt(Sigma_off)
        sqrt_Sigma_on = np.sqrt(Sigma_on)
        args_list = []
        for i, row in enumerate(ds_all.itertuples(index=False)):
            args_list.append(
                (
                    i,
                    row.CENTRIC,
                    row.loc_off,
                    row.scale_off,
                    row.loc_on,
                    row.scale_on,
                    sqrt_eps_arr[i],
                    sqrt_Sigma_on[i],
                    sqrt_Sigma_off[i],
                    p,
                    eps,
                    row.low_off,
                    row.high_off,
                    r,
                )
            )
    else:
        I_off = ds_all.I_off.to_numpy()
        SigI_off = ds_all.SigI_off.to_numpy()
        I_on = ds_all.I_on.to_numpy()
        SigI_on = ds_all.SigI_on.to_numpy()
        Sigma_off = mean_intensity_by_resolution(
            I_off / multiplicity, ds_all.dHKL.to_numpy()
        )
        Sigma_on = mean_intensity_by_resolution(
            I_on / multiplicity, ds_all.dHKL.to_numpy()
        )
        sqrt_Sigma_off = np.sqrt(Sigma_off)
        sqrt_Sigma_on = np.sqrt(Sigma_on)
        args_list = []
        for i, row in enumerate(ds_all.itertuples(index=False)):
            args_list.append(
                (
                    i,
                    row.CENTRIC,
                    I_off[i],
                    SigI_off[i],
                    I_on[i],
                    SigI_on[i],
                    Sigma_off[i],
                    Sigma_on[i],
                    p,
                    eps,
                    sqrt_eps_arr[i],
                    sqrt_Sigma_on[i],
                    sqrt_Sigma_off[i],
                    r,
                )
            )

    num_procs = args.nproc if args.nproc is not None else mp.cpu_count()

    worker = (
        estimate_reflection_intensity if args.use_intensities else estimate_reflection
    )

    with mp.Pool(
        processes=num_procs,
        initializer=init_shared_memory,
        initargs=(
            GS_ac_shm_obj.name,
            ES_ac_shm_obj.name,
            GS_c_shm_obj.name,
            ES_c_shm_obj.name,
            nsamples,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(worker, args_list, chunksize=200),
                total=len(args_list),
                disable=args.disable_progress_bar,
            )
        )

    # Collect results
    ES_abs_2_array = np.full(len(ds_all), np.nan)
    SIGES_abs_2_array = np.full(len(ds_all), np.nan)
    FS_abs_2_array = np.full(len(ds_all), np.nan)
    SIGFS_abs_2_array = np.full(len(ds_all), np.nan)
    loglik_array = np.full(len(ds_all), np.nan, dtype=np.float64)

    for i, es_val, es_sig, fs_val, fs_sig, ll_i in results:
        ES_abs_2_array[i] = es_val
        SIGES_abs_2_array[i] = es_sig
        FS_abs_2_array[i] = fs_val
        SIGFS_abs_2_array[i] = fs_sig
        loglik_array[i] = ll_i

    total_nll = -np.sum(loglik_array)

    # Assign and cast to MTZ-friendly types
    ds_all["ES_abs_2"] = ES_abs_2_array.astype("float32")
    ds_all["SIGES_abs_2"] = SIGES_abs_2_array.astype("float32")
    ds_all["FS_abs_2"] = FS_abs_2_array.astype("float32")
    ds_all["SIGFS_abs_2"] = SIGFS_abs_2_array.astype("float32")

    for col, mtz_type in [
        ("ES_abs_2", "F"),
        ("SIGES_abs_2", "Q"),
        ("FS_abs_2", "F"),
        ("SIGFS_abs_2", "Q"),
    ]:
        ds_all[col] = ds_all[col].astype(mtz_type)

    ds_all.dropna(inplace=True)
    # ds_all.infer_mtz_dtypes(inplace=True)
    ds_all[["ES_abs_2", "SIGES_abs_2", "FS_abs_2", "SIGFS_abs_2", "CENTRIC"]].write_mtz(
        args.outfile
    )

    # Cleanup shared memory
    for shm in [GS_ac_shm_obj, ES_ac_shm_obj, GS_c_shm_obj, ES_c_shm_obj]:
        shm.close()
        shm.unlink()

    return ds_all, total_nll


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    if args.default_scan:
        # fixed r

        if not args.rDW:
            args.rDW = 0.9

        # disallow conflicting options
        if args.factor or args.es_fraction:
            raise ValueError(
                "--default_scan cannot be used with --es-fraction or --factor"
            )

        p_values = np.arange(0.05, 0.51, 0.05)

        base_out = args.outfile

        best = (None, np.inf)  # (p, ll)
        scan_rows = []

        for p in p_values:
            args.es_fraction = float(p)

            args.outfile = base_out.replace(".mtz", f"_p{p:.2f}.mtz")

            print(f"Running default scan: r={args.rDW}, p={p:.2f}")
            ds_out, total_nll = extrapolate_dw(args)
            print(f"Negative Log Likelihood = {total_nll}")

            scan_rows.append((p, total_nll))

            if np.isfinite(total_nll) and total_nll < best[1]:
                best = (p, total_nll)

        if best[0] is None:
            raise RuntimeError("No finite NLL values found in scan.")
        print("\nDefault scan MLE (grid):")
        print(f"  r={args.rDW}, p={best[0]:.2f}, NLL={best[1]:.3f}")
    else:
        ds_out, total_nll = extrapolate_dw(args)
        print(f"NLL = {total_nll}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-on",
        "--onmtz",
        nargs="+",
        required=True,
        help=".mtz file for perturbed dataset",
    )
    parser.add_argument(
        "-off",
        "--offmtz",
        nargs="+",
        required=True,
        help=".mtz file for ground state dataset",
    )
    parser.add_argument(
        "-use_SF",
        "--use_structure_factors",
        nargs=2,
        metavar=("f_col, sigf_col"),
        help="Use structure factors from French-Wilson scaling. Specified as (F, SigF)",
    )
    parser.add_argument(
        "-use_I",
        "--use_intensities",
        nargs=2,
        metavar=("i_col, sigi_col"),
        help="Use integrated intensities. Specified as (I, SigI)",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1_000_000,
        help="Number of importance samples",
    )
    parser.add_argument(
        "-r",
        "--rDW",
        type=float,
        default=0.9,
        help="Double Wilson r (correlation) parameter",
    )
    parser.add_argument(
        "-p", "--es-fraction", type=float, help="Excited state fraction p"
    )
    parser.add_argument(
        "-f", "--factor", type=float, help="Extrapolation factor f = 1/p"
    )
    parser.add_argument(
        "-o", "--outfile", default="esf_dw.mtz", help="Output file name"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processors for multiprocessing",
    )
    parser.add_argument(
        "--default_scan",
        action="store_true",
        help="Run default scan with r=0.9 and p from 0.05 to 0.5 in steps of 0.05",
    )
    parser.add_argument("--disable-progress-bar", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument(
        "--seed",
        type=int,
        default=28,
        help="Random seed for generating Monte Carlo samples",
    )
    return parser


# For getting truncated Normal parameters using method of moments
def reparam(df):
    l = len(df["F"])
    high = np.repeat(np.array([1e10], dtype=np.float32), l)
    df["high"] = high
    low = np.repeat(np.array([1e-32], dtype=np.float32), l)
    df["low"] = low

    mean = df["F"].to_numpy()
    std = df["SigF"].to_numpy()
    locs = np.zeros(len(mean))
    scales = np.zeros(len(std))

    a = 1e-32
    b = 1e10

    for i in range(len(mean)):
        m = mean[i]
        s = std[i]

        guess_x = (a - m) / s
        guess_y = (b - m) / s
        sol = optimize.root(equations, x0=[guess_x, guess_y], args=(m, s))
        alpha_hat, beta_hat = sol.x
        sigma_hat = (a - b) / (alpha_hat - beta_hat)
        mu_hat = a - sigma_hat * alpha_hat
        sigma_hat = np.abs(sigma_hat)

        locs[i] = mu_hat
        scales[i] = sigma_hat

    df["loc"] = locs
    df["scale"] = scales
    df = df.infer_mtz_dtypes()

    return df


# Method of Moments equations for the reparam function
def equations(ab, m, s):
    a = 1e-32
    b = 1e10

    alpha, beta = ab
    Z = norm.cdf(beta) - norm.cdf(alpha)
    lam = (norm.pdf(alpha) - norm.pdf(beta)) / Z
    nu = 1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / Z - lam**2
    sigma = (b - a) / (beta - alpha)
    mu = a - sigma * alpha
    return [mu + sigma * lam - m, sigma**2 * nu - s**2]


# worker function for computing the log-mean in the likelihood caclulation
def _logmeanexp(logw):
    m = np.max(logw)
    return m + np.log(np.mean(np.exp(logw - m) + 1e-300))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
