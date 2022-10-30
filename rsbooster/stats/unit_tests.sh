#!/bin/bash

# python rsplit.py ~/merge_eo_fast/thermolysin_xval_0.mtz -b 15
# python rsplit.py ~/PYP/merge_example/pyp-dw-test_xval_* -b 15
# python cchalf.py ~/merge_eo_fast/thermolysin_xval_0.mtz -b 15 -m pearson
# python cchalf.py ~/PYP/merge_example/pyp-dw-test_xval_* -b 15 -m spearman
# python ccanom.py ~/merge_eo_fast/thermolysin_xval_0.mtz -b 15 -m spearman
python ccpred.py -i ~/PYP/merge_example/pyp-dw-test_predictions_* -b 15 -m spearman
# python careless_report.py -d ~/PYP/merge_example/ -p pyp-dw-test
