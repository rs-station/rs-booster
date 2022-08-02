from setuptools import setup, find_packages


# Get version number
def getVersionNumber():
    with open("rsbooster/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

DESCRIPTION = "Useful scripts for analyzing diffraction"
LONG_DESCRIPTION = """
rs-booster contains commandline scripts for diffraction data analysis tasks.

This package can be viewed as a "booster pack" for reciprocalspaceship.
"""
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Hekstra-Lab/rs-booster/issues",
    "Source Code": "https://github.com/Hekstra-Lab/rs-booster",
}


setup(
    name="rs-booster",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    license="MIT",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Jack B. Greisman",
    author_email="greisman@g.harvard.edu",
    url="https://github.com/Hekstra-Lab/rs-booster",
    project_urls=PROJECT_URLS,
    python_requires=">3.7",
    install_requires=["reciprocalspaceship", "matplotlib", "seaborn"],
    entry_points={
        "console_scripts": [
            "rs.extrapolate=rsbooster.esf.extrapolate:main",
            "rs.scaleit=rsbooster.scaleit.scaleit:main",
            "rs.internal_diffmap=rsbooster.diffmaps.internaldiffmap:main",
            "rs.ccsym=rsbooster.stats.ccsym:main",
            "rs.ccanom=rsbooster.stats.ccanom:main",
            "rs.cchalf=rsbooster.stats.cchalf:main",
            "rs.ccpred=rsbooster.stats.ccpred:main",
            "rs.diffmap=rsbooster.diffmaps.diffmap:main",
            "rs.precog2mtz=rsbooster.io.precog2mtz:main",
            "rs.find_peaks=rsbooster.realspace.find_peaks:find_peaks",
            "rs.find_difference_peaks=rsbooster.realspace.find_peaks:find_difference_peaks",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
    ],
)
