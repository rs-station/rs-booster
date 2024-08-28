#!/usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import glob


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
        help="unit cell params (default will be average experiment crystal)",
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="show some stdout")
    parser.add_argument("--extra-cols", dest="extra_cols", nargs="+", type=str, default=None, help="attemp to pull in additional columns")
    parser.add_argument("--ext", type=str, default="integrated.refl", help="read files with this extension")
    parser.add_argument("--tag", type=str, default=None, help="only select files containing this string")
    return parser


def print_refl():
    parser = ArgumentParser()
    parser.add_argument("reflfile", type=str, help="path to a integrated.refl file")
    args = parser.parse_args()
    from reciprocalspaceship.io import print_refl_info
    print_refl_info(args.reflfile)


def _write(ds, mtzname):
    """write the RS dataset to mtz file"""
    ds.write_mtz(mtzname)


def get_fnames(dirnames, verbose=False, tag=None, ext="integrated.refl"):
    """

    Parameters
    ----------
    dirnames: list of str, folders to search for files
    verbose: bool, whether to print stdout
    tag: str, only select files whose names contain this string
    ext: str, only select files ending with this string

    Returns
    -------
    list of filenames
    """
    fnames = []
    for dirname in dirnames:
        fnames += glob.glob(dirname + f"/*{ext}")
    if verbose:
        print(f"Found {len(fnames)} files")
    if tag is not None:
        fnames = [f for f in fnames if tag in f]
        if verbose:
            print(f"Selected {len(fnames)} files with {tag} in the name.")
    return fnames


def ray_main():
    parser = get_parser()
    parser.add_argument("--numjobs", default=10, type=int, help="number of workers!")
    args = parser.parse_args()
    assert args.ucell is not None
    assert args.symbol is not None

    fnames = get_fnames(args.dirnames, args.verbose, tag=args.tag, ext=args.ext)
    ds = read_dials_stills(fnames, unitcell=args.ucell, spacegroup=args.symbol, numjobs=args.numjobs,
                           parallel_backend="ray", extra_cols=args.extra_cols, verbose=args.verbose)
    _write(ds, args.mtz)


def mpi_main():
    parser = get_parser()
    args = parser.parse_args()
    assert args.ucell is not None
    assert args.symbol is not None
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    fnames = get_fnames(args.dirnames, args.verbose, tag=args.tag, ext=args.ext)
    ds = read_dials_stills(fnames, unitcell=args.ucell, spacegroup=args.symbol, parallel_backend="mpi",
                           extra_cols=args.extra_cols, verbose=args.verbose)
    if COMM.rank == 0:
        _write(ds, args.mtz)


if __name__ == "__main__":
    ray_main()
