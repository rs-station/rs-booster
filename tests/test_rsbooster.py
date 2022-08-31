import rsbooster


def test_version():
    """
    Test rsbooster.getVersionNumber() method exists and gives same result as
    rsbooster.__version__
    """
    assert rsbooster.getVersionNumber() == rsbooster.__version__
