from setuptools import setup, find_packages

setup(
    name="efxtools",
    version="0.0.2",
    author="Jack B. Greisman",
    author_email="greisman@g.harvard.edu",
    packages=find_packages(),
    description="",
    install_requires=["reciprocalspaceship", "matplotlib", "seaborn"],
    entry_points={
        "console_scripts": [
            "efxtools.extrapolate=efxtools.esf.extrapolate:main",
            "efxtools.scaleit=efxtools.scaleit.scaleit:main",
            "efxtools.internal_diffmap=efxtools.diffmaps.internaldiffmap:main",
            "efxtools.ccsym=efxtools.stats.ccsym:main",
            "efxtools.diffmap=efxtools.diffmaps.diffmap:main",
        ]
    },
)
