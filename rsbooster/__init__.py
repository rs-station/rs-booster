# Version number for rs-booster
def getVersionNumber():
    version = None
    try:
        from setuptools.version import metadata

        version = metadata.version("rs-booster")
    except ImportError:
        from setuptools.version import pkg_resources

        version = pkg_resources.require("rs-booster")[0].version

    return version



__version__ = getVersionNumber()
