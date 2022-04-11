#!/usr/bin/env python
from argparse import ArgumentParser
import reciprocalspaceship as rs


def parse_arguments():
    desc = """Convert precognition ingegration results to `.mtz` files for mergning in Careless."""

    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "--remove-sys-absences",
        action="store_true",
        default=False,
        help="Optionally remove systematic absences from the data according to --spacegroup or --spacegroup-for-absences if supplied.",
    )
    parser.add_argument(
        "--spacegroup-for-absences",
        type=str,
        default=None,
        help="Optionally use a different spacegroup to compute systematic absences. This may be useful for some EF-X data.",
    )
    parser.add_argument(
        "--spacegroup", type=str, required=True, help="The spacegroup of the data"
    )
    parser.add_argument(
        "--cell",
        type=float,
        required=True,
        nargs=6,
        help="The unit cell supplied as six floats. "
        "For example, --spacegroup 34. 45. 98. 90. 90. 90.",
    )
    parser.add_argument(
        "ii_in",
        nargs="+",
        type=str,
        help="Precognition `.ii` file(s)",
    )
    parser.add_argument(
        "-o",
        "--mtz-out",
        type=str,
        default="integrated.mtz",
        help="Name of the output mtz file.",
    )
    parser = parser.parse_args()
    return parser


def make_dataset(filenames, spacegroup, cell):
    """
    Make an rs.DataSet from all *.ii the files in filenames.

    Parameters
    ----------
    filenames : list or tuple
        List or tuple of strings corresponding to precognition `ii` files.
    spacegroup : gemmi.SpaceGroup or similar
    cell : gemmi.UnitCell or similar

    Returns
    -------
    dataset : rs.DataSet
        Dataset containing the Precognition Laue data from filenames
    """
    datasets = []
    for i, f in enumerate(sorted(filenames), 1):
        ds = rs.read_precognition(f, spacegroup=spacegroup, cell=cell)
        ds["BATCH"] = i
        ds["BATCH"] = ds["BATCH"].astype(rs.BatchDtype())
        datasets.append(ds)
    return rs.concat(datasets)


def main():
    parser = parse_arguments()

    # Parse the output filename(s)
    if isinstance(parser.ii_in, str):
        filenames = [parser.ii_in]
    else:
        filenames = parser.ii_in

    # Parse simple arguments
    cell = parser.cell
    spacegroup = parser.spacegroup
    outfile = parser.mtz_out

    ds = make_dataset(filenames, spacegroup, cell)

    if parser.remove_sys_absences:
        sys_absences_spacegroup = parser.spacegroup_for_absences
        if sys_absences_spacegroup is None:
            sys_absences_spacegroup = spacegroup
        ds.spacegroup = sys_absences_spacegroup
        ds.remove_absences(inplace=True)
        ds.spacegroup = spacegroup

    ds.write_mtz(outfile)


if __name__ == "__main__":
    main()
