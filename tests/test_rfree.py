from rsbooster.utils.rfree import rfree
import pytest
import numpy as np


def test_rfree():
    """
    Test that "seed" parameter creates reproducible rfree flags, whereas omitting it does not
    """

    params = {
        "cell": (30, 40, 50, 90, 90, 90),
        "sg": 19,
        "dmin": 1.5,
        "rfraction": 0.05,
    }

    flags = rfree(**params, seed=2022)
    flags_same_seed = rfree(**params, seed=2022)
    flags_different_seed = rfree(**params, seed=None)

    assert all(flags == flags_same_seed)
    assert any(flags != flags_different_seed)

    return


@pytest.mark.parametrize("rfraction", [0.01, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("dmin", [1.8, 2.0, 3.0])
def test_rfree_fraction(rfraction, dmin, cell=(30, 40, 90, 90, 90, 90), sg=19, seed=None):

    flags = rfree(cell=cell, sg=sg, dmin=dmin, rfraction=rfraction, seed=seed)

    x = flags["R-free-flags"].to_numpy()
    assert abs(x.sum() - len(x) * rfraction) / len(x) <= 0.02

    return
