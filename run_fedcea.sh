#!/bin/bash

# Source the conda configuration
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate flower

# Run the Python script with a parameter
python ./fedcea_main.py -alpha 0.5

