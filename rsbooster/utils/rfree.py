#!/usr/bin/env python
import argparse
import reciprocalspaceship as rs


def rfree(cell, sg, dmin, rfraction, seed):

    h, k, l = rs.utils.generate_reciprocal_asu(cell, sg, dmin).T

    ds = (
        rs.DataSet(
            {
                "H": h,
                "K": k,
                "L": l,
            },
            cell=cell,
            spacegroup=sg,
        )
        .infer_mtz_dtypes()
        .set_index(["H", "K", "L"])
    )

    ds = rs.utils.add_rfree(ds, rfraction, seed=seed)

    return ds


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create an mtz containing rfree flags",
    )
    # Required arguments
    parser.add_argument(
        "-o", "--outfile", default="rfree.mtz", help="Output MTZ filename"
    )

    parser.add_argument(
        "-f",
        "--from-file",
        default=None,
        type=str,
        help=(
            "Use the cell and spacegroup from the specified mtz file. "
            "Either this or `--cell` and `--spacegroup` must be provided. "
            "If no `--dmin` is provided, dmin will be inferred from this file."
        ),
    )

    parser.add_argument(
        "-c",
        "--cell",
        nargs=6,
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        type=float,
        default=None,
        help=(
            "Cell for output mtz file containing rfree flags. Specified as (a, b, c, alpha, beta, gamma)"
        ),
    )

    parser.add_argument(
        "-sg",
        "--spacegroup",
        type=str,
        default=None,
        help=("Spacegroup for output mtz file containing rfree flags"),
    )

    parser.add_argument(
        "-d",
        "--dmin",
        type=float,
        help=("Maximum resolution of reflections to be included"),
    )

    parser.add_argument(
        "-r",
        "--rfraction",
        required=True,
        type=float,
        help=("Fraction of reflections to be flagged as Rfree"),
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help=("Seed to random number generator for reproducible Rfree flags"),
    )

    return parser#.parse_args() # making docs works best when a function returns just the parser


def main():
    args = parse_arguments().parse_args()

    msg = "Either --from-file or both --cell and --spacegroup must be specified"
    if args.from_file is None:
        if args.cell is None or args.spacegroup is None:
            raise ValueError(msg)
    else:
        if args.cell is not None or args.spacegroup is not None:
            raise ValueError(msg)
        ds = rs.read_mtz(args.from_file)
        args.cell = ds.cell
        args.spacegroup = ds.spacegroup
        
        if args.dmin is None:
            args.dmin = ds.compute_dHKL()["dHKL"].min()

    flags = rfree(
        args.cell, args.spacegroup, args.dmin, args.rfraction, args.seed
    )

    flags.write_mtz(args.outfile)

    return


if __name__ == "__main__":
    main()
