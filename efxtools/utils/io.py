import reciprocalspaceship as rs


def subset_to_FSigF(mtzpath, data_col, sig_col, column_names_dict={}):
    """
    Utility function for reading MTZ and returning DataSet with F and SigF.

    Parameters
    ----------
    mtzpath : str, filename
        Path to MTZ file to read
    data_col : str, column name
        Column name for data column. If Intensity is specified, it will be
        French-Wilson'd.
    sig_col : str, column name
        Column name for sigma column. Must select for a StandardDeviationDtype.
    column_names_dict : dictionary
        If particular column names are desired for the output, this can be specified
        as a dictionary that includes `data_col` and `sig_col` as keys and what
        values they should map to.

    Returns
    -------
    rs.DataSet
    """
    mtz = rs.read_mtz(mtzpath)

    # Check dtypes
    if not isinstance(
        mtz[data_col].dtype, (rs.StructureFactorAmplitudeDtype, rs.IntensityDtype)
    ):
        raise ValueError(
            f"{data_col} must specify an intensity or |F| column in {mtzpath}"
        )
    if not isinstance(mtz[sig_col].dtype, rs.StandardDeviationDtype):
        raise ValueError(
            f"{sig_col} must specify a standard deviation column in {mtzpath}"
        )

    # Run French-Wilson if intensities are provided
    if isinstance(mtz[data_col].dtype, rs.IntensityDtype):
        scaled = rs.algorithms.scale_merged_intensities(
            mtz, data_col, sig_col, mean_intensity_method="anisotropic"
        )
        mtz = scaled.loc[:, ["FW-F", "FW-SIGF"]]
        mtz.rename(columns={"FW-F": data_col, "FW-SIGF": sig_col}, inplace=True)
    else:
        mtz = mtz.loc[:, [data_col, sig_col]]

    mtz.rename(columns=column_names_dict, inplace=True)
    return mtz
