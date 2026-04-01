import numpy as np
import pytest

import gemmi
import reciprocalspaceship as rs

RESOLUTION = 1.0  # TODO make this 0.8 so some are OOR
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=12.0, alpha=90, beta=90, gamma=120)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P63")


@pytest.fixture(scope="session")
def np_rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture
def random_dataset(np_rng: np.random.Generator) -> rs.DataSet:
    hall = rs.utils.generate_reciprocal_asu(UNIT_CELL, SPACE_GROUP, RESOLUTION, anomalous=False)
    sigma = 1.0
    number_of_reflections = hall.shape[0]

    ds = rs.DataSet(
        {
            "H": hall[:, 0],
            "K": hall[:, 1],
            "L": hall[:, 2],
            "IMEAN": sigma * np_rng.normal(size=number_of_reflections),
            "SIGIMEAN": sigma * np.ones(number_of_reflections),
        },
        spacegroup=SPACE_GROUP,
        cell=UNIT_CELL,
    ).infer_mtz_dtypes()

    ds = ds.set_index(["H", "K", "L"])
    ds["IMEAN"] = ds["IMEAN"].astype(rs.IntensityDtype())
    ds["SIGIMEAN"] = ds["SIGIMEAN"].astype(rs.StandardDeviationDtype())

    return ds