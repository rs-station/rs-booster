# rs-booster
`rs-booster` contains commandline scripts for diffraction data analysis tasks.

This package can be viewed as a "booster rocket" for [`reciprocalspaceship`](https://github.com/Hekstra-Lab/reciprocalspaceship).


### Installation

The fastest way to install `rs-booster` is using pip:

```bash
pip install rs-booster
```

If you are interested in getting access to new features that haven't yet made it into a release, you can install `rs-booster` from source:

```bash
git clone https://github.com/Hekstra-Lab/rs-booster.git
cd rs-booster
python -m pip install -e .
```

### Design and usage

`rs-booster` is designed primarily as a command line interface. 
Applications from this package are prefixed with `rs.`.

Users can **list available commands** by typing `rs.` and double-pressing the `TAB` key. 
Each subprogram is documented using the [argparse library](https://docs.python.org/3/library/argparse.html).
To get usage info for a subprogram, use the `-h` or `--help` flag. 
For instance,

```bash
$ rs.find_peaks -h
```

will print the following

```bash
usage: rs.find_peaks [-h] -f STRUCTURE_FACTOR_KEY -p PHASE_KEY [-o CSV_OUT] [-z SIGMA_CUTOFF] [-w WEIGHT_KEY] [--sample-rate SAMPLE_RATE] [--min-volume MIN_VOLUME]
                           [--min-score MIN_SCORE] [--min-peak MIN_PEAK] [-d DISTANCE_CUTOFF] [--use-long-names]
                           mtz_file pdb_file

Search an electron density map for peaks in the vicinity of a structure.

positional arguments:
  mtz_file
  pdb_file

options:
  -h, --help            show this help message and exit
  -f STRUCTURE_FACTOR_KEY, --structure-factor-key STRUCTURE_FACTOR_KEY
                        column label of the structure factor you want to use.
  -p PHASE_KEY, --phase-key PHASE_KEY
                        column label of the phase you want to use.
  -o CSV_OUT, --csv-out CSV_OUT
                        output the report to a csv file
  -z SIGMA_CUTOFF, --sigma-cutoff SIGMA_CUTOFF
                        the z-score cutoff for voxels to be included in the peak search. the default is 1.5
  -w WEIGHT_KEY, --weight-key WEIGHT_KEY
                        column label of any weights you wish to apply to the map.
  --sample-rate SAMPLE_RATE
                        change fft oversampling from the default (3).
  --min-volume MIN_VOLUME
                        the minimum volume of peaks with default zero.
  --min-score MIN_SCORE
                        the minimum score of peaks with default zero.
  --min-peak MIN_PEAK   the minimum peak value with default zero.
  -d DISTANCE_CUTOFF, --distance-cutoff DISTANCE_CUTOFF
                        the distance cutoff of nearest neighbor search with default of 4 angstroms.
  --use-long-names      use more verbose column names in the peak report.

```

