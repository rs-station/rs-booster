import warnings
import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import gemmi



long_names = {
    "chain"   : "Chain",
    "seqid"   : "SeqID",
    "residue" : "Residue",
    "name"    : "Atom Name",
    "dist"    : "Dist (Å)",
    "peak"    : "Peak Value",
    "peakz"   : "Peak Value (Z-score)",
    "score"   : "Peak Score",
    "scorez"  : "Peak Score (Z-score)",
    "cenx"    : "Centroid (x)",
    "ceny"    : "Centroid (y)",
    "cenz"    : "Centroid (z)",
    "coordx"  : "Coord (x)",
    "coordy"  : "Coord (y)",
    "coordz"  : "Coord (z)",
}


def peak_report(
        structure, 
        grid,
        sigma_cutoff,
        min_volume = 0.,
        min_score = 0.,
        min_peak = 0.,
        distance_cutoff = 4.,
        use_long_names = False,
        negate=False,
        sort_by_key='peakz',
    ):
    """
    Build a report summarizing peaks in a map which are in the vicinity of atoms in the structure.

    For example,

    ```python
    structure = gemmi.read_pdb(pdb_file_name)
    mtz = gemmi.read_mtz_file(mtz_file_name)
    grid = mtz.transform_f_phi_to_map(
        "ANOM", "PHANOM", sample_rate=3.0
    )

    report = peak_report(
        structure, grid, 
        sigma_cutoff=10.,
    )
    ```
    will find peaks in an anomalous difference map above 10 sigma. 

    For difference maps, it might make sense to use a pattern such as,

    ```python
    report = peak_report(structure, grid, sigma_cutoff=3.5),
    report = pd.concat((
        report,
        peak_report(structure, grid, sigma_cutoff=3.5, negate=True),
    ))
    ```
    which will find positive and negative difference map peaks above
    3.5 sigma and concatenate them into the same report. 


    Parameters
    ----------
    structure : gemmi.Structure
        The structure which will be searched for atoms near the electron density peaks. 
    grid : gemmi.FloatGrid
        This is an electron density map in the form of a gemmi.FloatGrid instance.
    sigma_cutoff : float
        The z-score cutoff at which pixels are thresholded for peak finding.
    min_volume : float (optional)
        The minimum volume of peaks which defaults to zero.
    min_score : float (optional)
        The minimum score of peaks which defaults to zero. See gemmi.find_blobs_by_flood_fill
    min_peak : float (optional)
        The minimum peak height of peaks which defaults to zero. See gemmi.find_blobs_by_flood_fill
    distance_cutoff : float (optional)
        This is the radius around atoms within which peaks will be kept. The default is 4 Angstroms.
        Making this number large may impact performance. 
    use_log_names : bool (optional)
        Optionally use more descriptive column names for the report. These may contain characters that
        make them less pleasant to work with in pandas. The default is False.
    negate : bool (optional)
        Optionally find peaks in the negative electron density. This can be useful for difference maps.
        The default is False.
    sort_by_key : str (optional)
        Sort report by values in this column. the "peakz" column is used by default.

    Returns
    -------
    peak_report : pd.DataFrame
        A dataframe summarizing the locations of found peaks and how they correspond to atoms in the structure.
    """

    if len(structure) > 1:
        warnings.warn(
            f"Multi-model PDBs are not supported. Using first model from file {pdb_file}.",
            UserWarning
        )

    cell = structure.cell
    model = structure[0]

    #Compute z-score cutoff
    mean,sigma = np.mean(grid),np.std(grid)
    cutoff = mean + sigma_cutoff * sigma

    #In gemmi peaks are blobs. So it goes.
    #This returns a list of `gemmi.Blob` objects
    blobs = gemmi.find_blobs_by_flood_fill(
        grid, 
        cutoff=cutoff, 
        min_volume=min_volume, 
        min_score=min_score, 
        min_peak=min_peak,
        negate=negate,
    )

    #This neighbor search object can find the atoms closest to query positions
    ns = gemmi.NeighborSearch(model, structure.cell, distance_cutoff).populate()

    peaks = []
    for blob in blobs:
        #This is a list of weird pointer objects. It is safest to convert them `gemmi.CRA` objects (see below)
        marks = ns.find_atoms(blob.centroid)
        if len(marks) == 0:
            continue

        cra = dist = None
        for mark in marks:
            image_idx = mark.image_idx
            _cra = mark.to_cra(model)
            _dist = cell.find_nearest_pbc_image(blob.centroid, _cra.atom.pos, mark.image_idx).dist()
            if cra is None:
                dist = _dist
                cra  = _cra
            elif _dist < dist:
                dist = _dist
                cra  = _cra

        record = {
            "chain"   :    cra.chain.name,
            "seqid"   :    cra.residue.seqid.num,
            "residue" :    cra.residue.name,
            "atom"    :    cra.atom.name,
            "element" :    cra.atom.element.name,
            "dist"    :    dist,
            "peakz"   :    (blob.peak_value-mean)/sigma,
            "scorez"  :    (blob.score-mean)/sigma,
            "peak"    :    blob.peak_value,
            "score"   :    blob.score,
            "cenx"    :    blob.centroid.x,
            "ceny"    :    blob.centroid.y,
            "cenz"    :    blob.centroid.x,
            "coordx"  :    cra.atom.pos.x,
            "coordy"  :    cra.atom.pos.y,
            "coordz"  :    cra.atom.pos.z,
        }
        if negate:
            negative_keys = ['peak', 'peakz', 'score', 'scorez']
            for k in negative_keys:
                record[k] = -record[k]
        peaks.append(record)

    out = pd.DataFrame.from_records(peaks)

    #In case there are no peaks we need to test the length
    if len(out) > 0:
        out = out.sort_values(sort_by_key, ascending=False)

    if use_long_names:
        out = out.rename(columns = long_names)
    return out

def parse_args(default_sigma_cutoff=1.5):
    from argparse import ArgumentParser

    program_description = """
    Search an electron density map for
    peaks in the vicinity of a structure.
    """
    parser = ArgumentParser(description=program_description)

    # Required / Common options
    parser.add_argument("-f", "--structure-factor-key", type=str, 
        required=True, help="column label of the structure factor you want to use.")
    parser.add_argument("-p", "--phase-key", type=str, 
        required=True, help="column label of the phase you want to use.")
    parser.add_argument("mtz_file")
    parser.add_argument("pdb_file")
    parser.add_argument("-o", "--csv-out", type=str, default=None, help="output the report to a csv file")
    parser.add_argument("-z", "--sigma-cutoff", required=False, default=default_sigma_cutoff, type=float, 
        help=f"the z-score cutoff for voxels to be included in the peak search. the default is {default_sigma_cutoff}")
    parser.add_argument("-w", "--weight-key", type=str, 
        required=False, default=None, help="column label of any weights you wish to apply to the map.")

    # More esoteric options
    parser.add_argument("--sample-rate", type=float, default=3.,
        help="change fft oversampling from the default (3).")
    parser.add_argument("--min-volume", type=float, default=0.,
        help="the minimum volume of peaks with default zero.")
    parser.add_argument("--min-score", type=float, default=0.,
        help="the minimum score of peaks with default zero.")
    parser.add_argument("--min-peak", type=float, default=0.,
        help="the minimum peak value with default zero.")
    parser.add_argument("-d", "--distance-cutoff", type=float, default=4.,
        help="the distance cutoff of nearest neighbor search with default of 4 angstroms.")
    parser.add_argument("--use-long-names", action='store_true',
        help="use more verbose column names in the peak report.")
    parser = parser.parse_args()
    return parser

def find_peaks():
    main(difference_map=False, default_sigma_cutoff=1.5)

def find_difference_peaks():
    main(difference_map=True, default_sigma_cutoff=3.0)

def main(difference_map=False, default_sigma_cutoff=1.5):
    parser = parse_args(default_sigma_cutoff)
    structure = gemmi.read_pdb(parser.pdb_file)
    ds = rs.read_mtz(parser.mtz_file)
    mtz = ds[[parser.phase_key]].copy()

    if parser.weight_key is not None:
        mtz[parser.structure_factor_key] = ds[parser.structure_factor_key] * ds[parser.weight_key]
    else:
        mtz[parser.structure_factor_key] = ds[parser.structure_factor_key] 

    mtz = mtz.to_gemmi()

    grid = mtz.transform_f_phi_to_map(
        parser.structure_factor_key, parser.phase_key, sample_rate=parser.sample_rate
    )

    out = peak_report(
        structure, grid, 
        sigma_cutoff=parser.sigma_cutoff,
        min_volume = parser.min_volume,
        min_score = parser.min_score,
        min_peak = parser.min_peak,
        distance_cutoff = parser.distance_cutoff,
        use_long_names = parser.use_long_names,
        negate=False,
    )
    if difference_map:
        out_neg = peak_report(
            structure, grid, 
            sigma_cutoff=parser.sigma_cutoff,
            min_volume = parser.min_volume,
            min_score = parser.min_score,
            min_peak = parser.min_peak,
            distance_cutoff = parser.distance_cutoff,
            use_long_names = parser.use_long_names,
            negate=True,
        )
        out = pd.concat((out, out_neg))

        #For difference maps re-sort the concatenated list
        peak_key = 'peakz' if 'peakz' in out else long_names['peakz']
        out['_sort_key'] = out[peak_key].abs()
        out = out.sort_values('_sort_key', ascending=False)
        del(out['_sort_key'])

    if parser.csv_out is not None:
        out.to_csv(parser.csv_out)

    print(out.to_csv())

