#!/usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import glob
import json
import os
from collections import Counter

import numpy as np
import gemmi


from reciprocalspaceship.io import read_dials_stills


def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dirnames",
        type=str,
        nargs="+",
        help="diffBragg.stills_process output folder(s) with *integrated.refls",
    )
    parser.add_argument("mtz", type=str, help="output mtz name")
    parser.add_argument(
        "--ucell",
        default=None,
        nargs=6,
        type=float,
        help="The six unit cell params: a (Ang.), b (Ang.), c (Ang.), alpha (deg.), beta (deg.), gamma (deg). If not provided, will attempt to find it in expt files that match the integrated refls. ",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Space group lookup symbol (e.g., P6522). If not provied, will attempt to find it in any expt files matching the integrated refls.")
    parser.add_argument("--verbose", action="store_true", help="show some stdout")
    parser.add_argument("--extra-cols", dest="extra_cols", nargs="+", type=str, default=None, help="attempt to pull in additional columns")
    parser.add_argument("--ext", type=str, default="integrated.refl", help="read files with this extension")
    parser.add_argument("--tag", type=str, default=None, help="only select files containing this string")
    return parser


def ucell_and_sg_from_expt(expt_fs, verbose=False):
    """expt_fs: list of dxtbx experiment list files written by stills_process"""
    params = []
    space_groups = []
    pi_to_deg = 180/np.pi
    for expt_f in expt_fs:
        xtals = json.load(open(expt_f, 'r'))['crystal']
        for xtal in xtals:
            a = xtal['real_space_a']
            b = xtal['real_space_b']
            c = xtal['real_space_c']
            ucell_a = np.linalg.norm(a)
            ucell_b = np.linalg.norm(b)
            ucell_c = np.linalg.norm(c)
            alpha = np.arccos(np.dot(b,c) / ucell_b / ucell_c)*pi_to_deg
            beta = np.arccos(np.dot(a,c) / ucell_a / ucell_c)*pi_to_deg
            gamma = np.arccos(np.dot(a,b) / ucell_a / ucell_b)*pi_to_deg
            params.append( (ucell_a, ucell_b, ucell_c, alpha, beta, gamma))

            hall = xtal["space_group_hall_symbol"]
            ops = gemmi.symops_from_hall(hall)
            sg_num = gemmi.find_spacegroup_by_ops(ops).number
            space_groups.append(sg_num)

    # unit cell is set as the median across crystals
    med_params = np.round(np.median(params,axis=0), 6)
    ucell = gemmi.UnitCell(*med_params)

    # space group is set as the most frequent across crystals
    sgs = Counter(space_groups)
    sg_num,_ = sgs.most_common(1)[0]
    sg = gemmi.SpaceGroup(sg_num)
    if verbose:
        print(f"Consensus unit cell from {len(params)} expts:", ucell)
        print(f"Consensus space group from {len(space_groups)} expts:", sg)

    return ucell, sg


def print_refl():
    parser = ArgumentParser()
    parser.add_argument("reflfile", type=str, help="path to a integrated.refl file")
    args = parser.parse_args()
    from reciprocalspaceship.io import print_refl_info
    print_refl_info(args.reflfile)


def _write(ds, mtzname, verbose=False):
    """write the RS dataset to mtz file"""
    if verbose:
        print(f"Writing MTZ {mtzname} ...")
    ds.write_mtz(mtzname)
    if verbose:
        print("Done writing MTZ.")


def get_fnames(dirnames, verbose=False, optional_tag=None, ext="integrated.refl"):
    """

    Parameters
    ----------
    dirnames: list of str, folders to search for files
    verbose: bool, whether to print stdout
    optional_tag: str, only select files whose names contain this string
    ext: str, only select files ending with this string

    Returns
    -------
    list of filenames
    """
    fnames = []
    for dirname in dirnames:
        fnames += glob.glob(dirname + f"/*{ext}")
    if verbose:
        print(f"Found {len(fnames)} reflection files.")
    if optional_tag is not None:
        fnames = [f for f in fnames if optional_tag in f]
        if verbose:
            print(f"Selected {len(fnames)} files with {optional_tag} in the name.")
    if not fnames:
        raise IOError(f"No filenames were found for loading with dirnames={dirnames}, optional_tag={optional_tag}, and ext={ext}")
    return fnames


def _reconcile_ucell_and_sg(fnames, args, max_expt_fs=5, verbose=False):
    """
    fnames: list of *.refl files, for which matching *.expt files with crystal models might exist
    args: parsed cmd argumensts
    max_expt_fs: in the event that many experiment list files exists, only parse this many for unit cell and sg info
    returns: ucell and sg objects. User provided sg and unit cell are prioritized
    """
    expt_fs = []
    for f in fnames:
        expt_f = f.replace(".refl", ".expt")
        if os.path.exists(expt_f):
            expt_fs.append(expt_f)
    if verbose and expt_fs:
        print(f"Found {len(expt_fs)} corresponding exptlist files. Will use {max_expt_fs} of them to estimate ucell and space group.")
    expt_fs = expt_fs[:max_expt_fs]
    ucell = sg = None
    if expt_fs:
        try:
            ucell, sg = ucell_and_sg_from_expt(expt_fs, verbose=verbose)
        except Exception as err:
            print(str(err))
    if args.ucell is not None:
        ucell = gemmi.UnitCell(*args.ucell)
        if verbose:
            print("Will use user-provided unit cell:", ucell)
    if args.symbol is not None:
        sg = gemmi.SpaceGroup(args.symbol)
        if verbose:
            print("Will use user-provided space group:", sg)

    if ucell is None or sg is None:
        raise OSError("No experiment lists found, so provide --ucell and --sg explicitly as cmdline arguments.")

    return  ucell, sg


def ray_main():
    parser = get_parser()
    parser.add_argument("--numjobs", default=10, type=int, help="number of workers!")
    args = parser.parse_args()

    fnames = get_fnames(args.dirnames, args.verbose, optional_tag=args.tag, ext=args.ext)
    ucell, symbol = _reconcile_ucell_and_sg(fnames, args, verbose=args.verbose)

    ds = read_dials_stills(fnames, unitcell=ucell, spacegroup=symbol, numjobs=args.numjobs,
                           parallel_backend="ray", extra_cols=args.extra_cols, verbose=args.verbose, 
                           mtz_dtypes=True)
    _write(ds, args.mtz, args.verbose)


def mpi_main():
    parser = get_parser()
    args = parser.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    fnames = get_fnames(args.dirnames, args.verbose, optional_tag=args.tag, ext=args.ext)
    ucell, symbol = _reconcile_ucell_and_sg(fnames, args, verbose=args.verbose)
    ds = read_dials_stills(fnames, unitcell=ucell, spacegroup=symbol, parallel_backend="mpi",
                           extra_cols=args.extra_cols, verbose=args.verbose, comm=comm,
                           mtz_dtypes=True)
    if comm.rank == 0:
        _write(ds, args.mtz, args.verbose)


if __name__ == "__main__":
    ray_main()
