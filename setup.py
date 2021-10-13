from setuptools import setup, find_packages

setup(
    name="efxtools",
    version="0.0.1",
    author="Jack B. Greisman",
    author_email="greisman@g.harvard.edu",
    packages=find_packages(),
    description="",
    install_requires=[
        "reciprocalspaceship",
    ],
    entry_points={
        "console_scripts": [
            "efxtools.extrapolate=efxtools.esf.extrapolate:main",
        ]
    },
    
)
