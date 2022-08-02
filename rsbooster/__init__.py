# Version number for rsbooster
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("rs-booster")[0].version
    return version


__version__ = getVersionNumber()
