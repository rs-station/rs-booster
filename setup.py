from setuptools import setup, find_packages

setup(
    name="rs-booster",
    version="0.0.3",
    author="Jack B. Greisman",
    author_email="greisman@g.harvard.edu",
    packages=find_packages(),
    description="",
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
)
