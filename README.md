# efxtools
Efxtools contains scripts for some important diffraction data analysis tasks. 


### Installation

For users who don't want to [configure ssh access](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to GitHub, 
you may install from source without git with the following steps. 

 1) [download the latest source code](https://github.com/Hekstra-Lab/efxtools/archive/refs/heads/main.zip)
 2) Unpack the `.zip` file from (1) using your system's archive tool. You may do this in the file browser on most modern operating systems. 
 3) In a terminal, navigate to the extracted source code directory (for instance `cd /home/user/efxtools-main`).
 4) Type `python setup.py install` or `pip install -e .`. Use the latter if you would like to modify the source code in the future. 

### Design and usage

Efxtools is designed primarily as a command line interface. 
Applications from this package are prefixed with `efxtools.`.

Users can **list available commands** by typing `efxtools.` and double-pressing the `TAB` key. 
Each subprogram is documented using the [argparse library](https://docs.python.org/3/library/argparse.html).
To get usage info for a subprogram, use the `-h` or `--help` flag. 
For instance,

```bash
$ efxtools.find_peaks -h
```

will print the following

```bash
usage: efxtools.find_peaks [-h] -f STRUCTURE_FACTOR_KEY -p PHASE_KEY [-o CSV_OUT] [-z SIGMA_CUTOFF] [-w WEIGHT_KEY] [--sample-rate SAMPLE_RATE] [--min-volume MIN_VOLUME]
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

