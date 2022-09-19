#!/usr/bin/env python
import argparse
from re import S
import reciprocalspaceship as rs


def make_rfree(output_file, cell, sg, dmin, rfraction, seed):

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
    ds.write_mtz(output_file)

    return


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-o", "--outfile", default="complete_with_rfree.mtz", help="Output MTZ filename"
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

    make_rfree(args.outfile, args.cell, args.spacegroup, args.dmin, args.rfraction, args.seed)

    print("rfree does stuff")
    return


if __name__ == "__main__":
    main()
