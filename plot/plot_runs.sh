#!/bin/bash

# generate CSV-files
python3 get_csv.py ../logs parent

# generate Plot
Rscript plot_runs.r data/logs plot pdf 