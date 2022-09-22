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
        "-c",
        "--cell",
        nargs=6,
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        type=float,
        required=True,
        help=(
            "Cell for output mtz file containing rfree flags. Specified as (a, b, c, alpha, beta, gamma)"
        ),
    )

    parser.add_argument(
        "-sg",
        "--spacegroup",
        required=True,
        type=int,
        help=("Spacegroup for output mtz file containing rfree flags"),
    )

    parser.add_argument(
        "-d",
        "--dmin",
        required=True,
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    flags = rfree(
        args.cell, args.spacegroup, args.dmin, args.rfraction, args.seed
    )

    flags.write_mtz(args.outfile)

    return


if __name__ == "__main__":
    main()
