#!/usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import glob


from reciprocalspaceship.io import dials


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
    return parser


def _write(ds, mtzname):
    """write the RS dataset to mtz file"""
    ds.write_mtz(mtzname)


def get_fnames(dirnames, verbose=False):
    fnames = []
    for dirname in dirnames:
        fnames += glob.glob(dirname + "/*integrated.refl")
    if verbose:
        print("Found %d files" % len(fnames))
    return fnames


def _set_logger(verbose):
    level = logging.CRITICAL
    if verbose:
        level = logging.DEBUG

    for log_name in ("rs.io.dials", "ray"):
        logger = logging.getLogger(log_name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def ray_main():
    parser = get_parser()
    parser.add_argument("--numjobs", default=10, type=int, help="number of workers!")
    args = parser.parse_args()
    assert args.ucell is not None
    assert args.symbol is not None
    _set_logger(args.verbose)

    fnames = get_fnames(args.dirnames, args.verbose)
    ds = dials.read_dials_stills(fnames, unitcell=args.ucell, spacegroup=args.symbol, numjobs=args.numjobs)
    _write(ds, args.mtz)


def mpi_main():
    parser = get_parser()
    args = parser.parse_args()
    assert args.ucell is not None
    assert args.symbol is not None
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    from reciprocalspaceship.io.dials_mpi import read_dials_stills_mpi
    _set_logger(args.verbose)
    fnames = get_fnames(args.dirnames, args.verbose)
    ds = read_dials_stills_mpi(fnames, unitcell=args.ucell, spacegroup=args.symbol)
    if COMM.rank==0:
        _write(ds, args.mtz)


if __name__ == "__main__":
    ray_main()
