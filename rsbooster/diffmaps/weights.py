def compute_weights(df, sigdf, alpha=0.0):
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.

    Parameters
    ----------
    df : series-like or array-like
        Array of DeltaFs (difference structure factor amplitudes)
    sigdf : series-like or array-like
        Array of SigDeltaFs (uncertainties in difference structure factor amplitudes)
    """
    w = 1 + (sigdf ** 2 / (sigdf ** 2).mean()) + alpha * (df ** 2 / (df ** 2).mean())
    return w ** -1
