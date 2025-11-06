#!/usr/bin/env python
"""
Maximum-likelihood estimation of (r, p) parameters for double-Wilson extrapolation. To be used before with the dw_extrapolate script.


Notes
-----
- For careless outputs: pass .mtz's without additional arguments
- For French-Wilson scaled structure factors from other software (e.g., XDS):
    - Use the --use_structure_factors flag, specifying F/SigF column names
- For non French-Wilson scaled datasets (e.g., from Aimless), use integrated Intensities datasets:
    - Use the --use_intensities flag, specifying I/SigI column names
- Provide initial parameter estimates usin gthe --init_r, --init_p flags
- To prioritize efficiency, this method has default sample size of n=10,000
    - We also recommend only using a subset of reflections to perform inference (about a quarter of all reflections) has worked well for us; the number of reflections used can be specified using the --subset flag
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm, norm
from scipy import optimize
import reciprocalspaceship as rs
import multiprocessing as mp
from multiprocessing import shared_memory


raw_Z_ac_shm = None
raw_Z_c_shm = None
raw_Z_ac = None
raw_Z_c = None


def init_shared_memory(Z_ac_name, Z_c_name, nsamples):
    """Attach each worker to the shared standard normal samples."""
    global raw_Z_ac_shm, raw_Z_c_shm, raw_Z_ac, raw_Z_c
    raw_Z_ac_shm = shared_memory.SharedMemory(name=Z_ac_name)
    raw_Z_c_shm = shared_memory.SharedMemory(name=Z_c_name)
    raw_Z_ac = np.ndarray((nsamples, 4), dtype=np.float32, buffer=raw_Z_ac_shm.buf)
    raw_Z_c = np.ndarray((nsamples, 2), dtype=np.float32, buffer=raw_Z_c_shm.buf)


# worker function for computing the log-mean in the likelihood caclulation
def _logmeanexp(logw):
    m = np.max(logw)
    return m + np.log(np.mean(np.exp(logw - m) + 1e-300))


# worker function for computing the log likelihood using the truncated Normal parameterization
def loglike_reflection_SF(args):
    (
        centric,
        loc_off,
        scale_off,
        low_off,
        high_off,
        loc_on,
        scale_on,
        low_on,
        high_on,
        sqrt_eps,
        sqrt_Sig_on,
        sqrt_Sig_off,
        r,
        p,
    ) = args

    # Transform standard normals to DW prior
    L_ac = np.sqrt(0.5) * np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [r, 0, np.sqrt(1 - r**2), 0],
            [0, r, 0, np.sqrt(1 - r**2)],
        ],
        dtype=np.float64,
    )
    L_c = np.array([[1, 0], [r, np.sqrt(1 - r**2)]], dtype=np.float64)

    if not centric:
        E_ac = raw_Z_ac.dot(L_ac.T)
        GS = E_ac[:, 0] + 1j * E_ac[:, 1]
        ES = E_ac[:, 2] + 1j * E_ac[:, 3]
        ON = (1 - p) * GS + p * ES
        OF_abs = np.abs(GS)
        ON_abs = np.abs(ON)
    else:
        E_c = raw_Z_c.dot(L_c.T)
        GS = E_c[:, 0]
        ES = E_c[:, 1]
        ON = (1 - p) * GS + p * ES
        OF_abs = np.abs(GS)
        ON_abs = np.abs(ON)

    k = np.median(ON_abs) / (np.median(OF_abs) + 1e-12)

    x_off = sqrt_eps * sqrt_Sig_off * OF_abs
    x_on = sqrt_eps * sqrt_Sig_on * (ON_abs / (k + 1e-12))

    a_off, b_off = (low_off - loc_off) / (scale_off + 1e-30), (high_off - loc_off) / (
        scale_off + 1e-30
    )
    a_on, b_on = (low_on - loc_on) / (scale_on + 1e-30), (high_on - loc_on) / (
        scale_on + 1e-30
    )

    rv_off = truncnorm(a_off, b_off, loc=loc_off, scale=scale_off)
    rv_on = truncnorm(a_on, b_on, loc=loc_on, scale=scale_on)

    logw = rv_off.logpdf(x_off) + rv_on.logpdf(x_on)
    return _logmeanexp(logw)


# worker function for computing the log likelihood using the standard Normal parameterization
def loglike_reflection_I(args):
    (
        centric,
        I_off_obs,
        SigI_off,
        I_on_obs,
        SigI_on,
        Sigma_off,
        Sigma_on,
        sqrt_eps,
        r,
        p,
    ) = args

    L_ac = np.sqrt(0.5) * np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [r, 0, np.sqrt(1 - r**2), 0],
            [0, r, 0, np.sqrt(1 - r**2)],
        ],
        dtype=np.float64,
    )
    L_c = np.array([[1, 0], [r, np.sqrt(1 - r**2)]], dtype=np.float64)

    if not centric:
        E_ac = raw_Z_ac.dot(L_ac.T)
        GS = E_ac[:, 0] + 1j * E_ac[:, 1]
        ES = E_ac[:, 2] + 1j * E_ac[:, 3]
        ON = (1 - p) * GS + p * ES
        OF_abs = np.abs(GS)
        ON_abs = np.abs(ON)
    else:
        E_c = raw_Z_c.dot(L_c.T)
        GS = E_c[:, 0]
        ES = E_c[:, 1]
        ON = (1 - p) * GS + p * ES
        OF_abs = np.abs(GS)
        ON_abs = np.abs(ON)

    k = np.median(ON_abs) / (np.median(OF_abs) + 1e-12)
    ON_abs = ON_abs / (k + 1e-12)

    loc_OF = (OF_abs**2) * Sigma_off * (sqrt_eps**2)
    loc_ON = (ON_abs**2) * Sigma_on * (sqrt_eps**2)

    rv_off = norm(loc=loc_OF, scale=SigI_off)
    rv_on = norm(loc=loc_ON, scale=SigI_on)

    logw = rv_off.logpdf(I_off_obs) + rv_on.logpdf(I_on_obs)
    return _logmeanexp(logw)


# Reading Arguments


def parse_arguments():
    p = argparse.ArgumentParser(
        description="MLE of (r, p) for DW model with parallel Monte Carlo"
    )
    p.add_argument("--onmtz", required=True, help=".mtz file for perturbed dataset")
    p.add_argument("--offmtz", required=True, help=".mtz file for ground state dataset")
    p.add_argument(
        "--use_structure_factors",
        "-use_SF",
        nargs=2,
        metavar=("F_COL", "SIGF_COL"),
        help="Use FW‑scaled |F|/SigF columns, e.g., F SigF",
    )
    p.add_argument(
        "--use_intensities",
        "-use_I",
        nargs=2,
        metavar=("I_COL", "SIGI_COL"),
        help="Use I/SigI with Normal likelihood, e.g., I SigI",
    )
    p.add_argument(
        "--nsamples",
        "-n",
        type=int,
        default=10_000,
        help="Monte Carlo samples (default 1e6)",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processes (default: cpu_count)",
    )
    p.add_argument("--init_r", type=float, default=0.9, help="Initial guess for r")
    p.add_argument("--init_p", type=float, default=0.125, help="Initial guess for p")
    p.add_argument(
        "--bounds_r", type=float, nargs=2, default=[1e-6, 1-1e-6], help="Bounds for r"
    )
    p.add_argument(
        "--bounds_p", type=float, nargs=2, default=[1e-6, 1 - 1e-6], help="Bounds for p"
    )
    p.add_argument("--maxiter", type=int, default=50, help="Max optimizer iterations")
    p.add_argument("--seed", type=int, default=13, help="Random seed for MC samples")
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Optional number of reflections to randomly subsample for faster runs",
    )
    p.add_argument("--disable_progress_bar", action="store_true")
    p.add_argument(
        "--out", "-o", default="results.json", help="Where to write JSON results"
    )
    return p


# Helper Functions


def mean_intensity_by_resolution(I, dHKL, bins=50, gridpoints=None):
    I = np.array(I, dtype=np.float64)
    dHKL = np.array(dHKL, dtype=np.float64)
    if gridpoints is None:
        gridpoints = bins * 20
    X = dHKL**-2.0
    bw = (X.max() - X.min()) / bins
    grid = np.linspace(X.min(), X.max(), gridpoints)
    K = np.exp(-0.5 * ((X[:, None] - grid[None, :]) / bw) ** 2)
    K = K / K.sum(0)
    protos = I @ K
    bw = grid[1] - grid[0]
    K = np.exp(-0.5 * ((X[:, None] - grid[None, :]) / bw) ** 2)
    K = K / K.sum(1)[:, None]
    Sigma = K @ protos
    return Sigma


def reparam(df):
    """Method‑of‑moments reparametrization for truncated normal on |F|.
    Input df must contain columns F,SigF.
    Produces columns: low, high, loc, scale.
    """
    from scipy import optimize as _opt

    l = len(df["F"])
    high = np.repeat(np.array([1e10], dtype=np.float32), l)
    low = np.repeat(np.array([1e-32], dtype=np.float32), l)

    mean = df["F"].to_numpy()
    std = df["SigF"].to_numpy()
    locs = np.zeros(len(mean))
    scales = np.zeros(len(std))

    a = 1e-32
    b = 1e10

    for i in range(len(mean)):
        m = float(mean[i])
        s = float(std[i])
        guess_x = (a - m) / (s + 1e-12)
        guess_y = (b - m) / (s + 1e-12)
        sol = _opt.root(equations, x0=[guess_x, guess_y], args=(m, s))
        alpha_hat, beta_hat = sol.x
        sigma_hat = (a - b) / (alpha_hat - beta_hat)
        mu_hat = a - sigma_hat * alpha_hat
        sigma_hat = abs(sigma_hat)
        locs[i] = mu_hat
        scales[i] = sigma_hat

    df["low"] = low
    df["high"] = high
    df["loc"] = locs
    df["scale"] = scales
    df = df.infer_mtz_dtypes()
    return df


def equations(ab, m, s):
    a = 1e-32
    b = 1e10
    alpha, beta = ab
    Z = norm.cdf(beta) - norm.cdf(alpha)
    lam = (norm.pdf(alpha) - norm.pdf(beta)) / (Z + 1e-300)
    nu = 1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (Z + 1e-300) - lam**2
    sigma = (b - a) / (beta - alpha)
    mu = a - sigma * alpha
    return [mu + sigma * lam - m, sigma**2 * nu - s**2]


def build_dataset(args):
    ds_on = rs.read_mtz(args.onmtz)
    ds_off = rs.read_mtz(args.offmtz)

    if args.use_intensities:
        I_col, SigI_col = args.use_intensities
        ds_on = ds_on.rename(columns={I_col: "I", SigI_col: "SigI"}).dropna(
            subset=["I", "SigI"], how="any"
        )
        ds_off = ds_off.rename(columns={I_col: "I", SigI_col: "SigI"}).dropna(
            subset=["I", "SigI"], how="any"
        )
        ds_all = ds_off.merge(
            ds_on,
            left_index=True,
            right_index=True,
            suffixes=("_off", "_on"),
            check_isomorphous=False,
        )
        if args.subset is not None and args.subset > 0:
            n = int(min(args.subset, len(ds_all)))
            ds_all = ds_all.sample(n=n, random_state=args.seed)
        ds_all = ds_all.copy()
        ds_all.label_centrics(inplace=True)
        ds_all.compute_multiplicity(inplace=True)
        ds_all.compute_dHKL(inplace=True)
        multiplicity = ds_all.EPSILON.to_numpy()
        sqrt_eps_arr = np.sqrt(multiplicity)
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
        return (
            ds_all,
            sqrt_eps_arr,
            sqrt_Sigma_on,
            sqrt_Sigma_off,
            I_off,
            SigI_off,
            I_on,
            SigI_on,
        )

    # Structure factors path
    if args.use_structure_factors:
        F_col, SigF_col = args.use_structure_factors
        ds_on = ds_on.rename(columns={F_col: "F", SigF_col: "SigF"}).dropna(
            subset=["F", "SigF"], how="any"
        )
        ds_off = ds_off.rename(columns={F_col: "F", SigF_col: "SigF"}).dropna(
            subset=["F", "SigF"], how="any"
        )
        ds_on = reparam(ds_on)
        ds_off = reparam(ds_off)
    else:
        ds_on = ds_on.dropna(subset=["F", "SigF"], how="any")
        ds_off = ds_off.dropna(subset=["F", "SigF"], how="any")
        # Expect columns loc/scale/low/high to be present already; else reparam
        if not set(["loc", "scale", "low", "high"]).issubset(ds_on.columns):
            ds_on = reparam(ds_on)
        if not set(["loc", "scale", "low", "high"]).issubset(ds_off.columns):
            ds_off = reparam(ds_off)

    ds_all = ds_off.merge(
        ds_on,
        left_index=True,
        right_index=True,
        suffixes=("_off", "_on"),
        check_isomorphous=False,
    )
    if args.subset is not None and args.subset > 0:
        n = int(min(args.subset, len(ds_all)))
        ds_all = ds_all.sample(n=n, random_state=args.seed)
    ds_all = ds_all.copy()
    ds_all.label_centrics(inplace=True)
    ds_all.compute_multiplicity(inplace=True)
    ds_all.compute_dHKL(inplace=True)

    multiplicity = ds_all.EPSILON.to_numpy()
    sqrt_eps_arr = np.sqrt(multiplicity)

    Sigma_off = mean_intensity_by_resolution(
        (ds_all.F_off**2 / multiplicity).to_numpy(), ds_all.dHKL.to_numpy()
    )
    Sigma_on = mean_intensity_by_resolution(
        (ds_all.F_on**2 / multiplicity).to_numpy(), ds_all.dHKL.to_numpy()
    )
    sqrt_Sigma_off = np.sqrt(Sigma_off)
    sqrt_Sigma_on = np.sqrt(Sigma_on)

    return ds_all, sqrt_eps_arr, sqrt_Sigma_on, sqrt_Sigma_off


def objective_factory(
    args, ds_all, sqrt_eps_arr, sqrt_Sigma_on, sqrt_Sigma_off, extra=None
):
    use_I = args.use_intensities is not None

    if not use_I:
        static = []
        for i, row in enumerate(ds_all.itertuples(index=False)):
            static.append(
                (
                    bool(row.CENTRIC),
                    float(row.loc_off),
                    float(row.scale_off),
                    float(row.low_off),
                    float(row.high_off),
                    float(row.loc_on),
                    float(row.scale_on),
                    float(row.low_on),
                    float(row.high_on),
                    float(sqrt_eps_arr[i]),
                    float(sqrt_Sigma_on[i]),
                    float(sqrt_Sigma_off[i]),
                )
            )
    else:
        I_off, SigI_off, I_on, SigI_on = extra
        static = []
        for i, row in enumerate(ds_all.itertuples(index=False)):
            static.append(
                (
                    bool(row.CENTRIC),
                    float(I_off[i]),
                    float(SigI_off[i]),
                    float(I_on[i]),
                    float(SigI_on[i]),
                    float(sqrt_Sigma_off[i] ** 2),  # pass Sigma_off
                    float(sqrt_Sigma_on[i] ** 2),  # pass Sigma_on
                    float(sqrt_eps_arr[i]),
                )
            )

    num_procs = args.nproc if args.nproc is not None else mp.cpu_count()

    pool = mp.Pool(
        processes=num_procs,
        initializer=init_shared_memory,
        initargs=(
            objective_factory.Z_ac_name,
            objective_factory.Z_c_name,
            args.nsamples,
        ),
    )

    worker = loglike_reflection_I if use_I else loglike_reflection_SF

    def objective(theta):
        r, p = float(theta[0]), float(theta[1])
        if not use_I:
            args_list = [s + (r, p) for s in static]
        else:
            args_list = [s + (r, p) for s in static]
        parts = list(
            tqdm(
                pool.imap(worker, args_list, chunksize=200),
                total=len(args_list),
                disable=args.disable_progress_bar,
                leave=False,
            )
        )
        total_ll = float(np.sum(parts))
        return -total_ll

    objective._pool = pool
    return objective


objective_factory.Z_ac_name = None
objective_factory.Z_c_name = None


def main():
    args = parse_arguments().parse_args()
    mp.set_start_method("spawn", force=True)

    rng = np.random.default_rng(args.seed)
    raw_Z_ac_local = rng.standard_normal(size=(args.nsamples, 4), dtype=np.float32)
    raw_Z_c_local = rng.standard_normal(size=(args.nsamples, 2), dtype=np.float32)

    Z_ac_shm_obj = shared_memory.SharedMemory(create=True, size=raw_Z_ac_local.nbytes)
    Z_c_shm_obj = shared_memory.SharedMemory(create=True, size=raw_Z_c_local.nbytes)
    np.ndarray(
        raw_Z_ac_local.shape, dtype=raw_Z_ac_local.dtype, buffer=Z_ac_shm_obj.buf
    )[:] = raw_Z_ac_local
    np.ndarray(raw_Z_c_local.shape, dtype=raw_Z_c_local.dtype, buffer=Z_c_shm_obj.buf)[
        :
    ] = raw_Z_c_local

    objective_factory.Z_ac_name = Z_ac_shm_obj.name
    objective_factory.Z_c_name = Z_c_shm_obj.name

    if args.use_intensities:
        (
            ds_all,
            sqrt_eps_arr,
            sqrt_Sigma_on,
            sqrt_Sigma_off,
            I_off,
            SigI_off,
            I_on,
            SigI_on,
        ) = build_dataset(args)
        extra = (I_off, SigI_off, I_on, SigI_on)
    else:
        ds_all, sqrt_eps_arr, sqrt_Sigma_on, sqrt_Sigma_off = build_dataset(args)
        extra = None

    objective = objective_factory(
        args, ds_all, sqrt_eps_arr, sqrt_Sigma_on, sqrt_Sigma_off, extra
    )

    x0 = np.array([args.init_r, args.init_p], dtype=float)
    bounds = [tuple(args.bounds_r), tuple(args.bounds_p)]

    def callback(theta):
        r, p = theta
        current_nll = objective(theta)
        print(f"[iteration] r={r:.5f}, p={p:.5f}, NLL={current_nll:.4f}")

    res = optimize.minimize(
        fun=objective,
        x0=x0,
        method="L-BFGS-B",
        callback=callback,
        bounds=bounds,
        options={"maxiter": args.maxiter, "disp": False},
    )

    # Close pool cleanly
    try:
        objective._pool.close()
        objective._pool.join()
    finally:
        pass

    # Cleanup shared memory
    for shm in [Z_ac_shm_obj, Z_c_shm_obj]:
        shm.close()
        shm.unlink()

    result = {
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "njev": int(res.njev) if getattr(res, "njev", None) is not None else None,
        "fun": float(res.fun),  # negative log-likelihood at optimum
        "r": float(res.x[0]),
        "p": float(res.x[1]),
        "bounds": {"r": args.bounds_r, "p": args.bounds_p},
        "init": {"r": args.init_r, "p": args.init_p},
        "nsamples": int(args.nsamples),
        "nproc": int(args.nproc) if args.nproc is not None else int(mp.cpu_count()),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
