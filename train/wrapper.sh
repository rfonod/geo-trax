#!/bin/bash -l
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)
#
# SLURM job submission wrapper for running training and inference scripts on HPC clusters.
#
# Usage:
#   sbatch train/wrapper.sh <script> [ARGS...]
#
# Examples:
#   sbatch train/wrapper.sh train/train.sh -m 8 -e 1 -s s
#
# Notes:
#   - Update --chdir to the absolute path of your geo-trax project root.
#   - Update the conda environment name if it differs from 'geo-trax'.
#   - Adjust --cpus-per-task, --mem, --time, and --gres to match your cluster's resources.

#SBATCH --job-name GeoTrax-Train
#SBATCH --output=R-%x.%j.out
#SBATCH --chdir /path/to/geo-trax
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 128G
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1

echo STARTING AT $(date)
echo "Job run at: $(hostname)"

#-------------------------------------------------------------------------------
# Activate conda virtual environment
#-------------------------------------------------------------------------------
echo -e "\nActivating conda environment..."
eval "$(conda shell.bash hook)"
conda activate geo-trax  # update to match your conda environment name
echo -e "Activated virtual environment: $CONDA_DEFAULT_ENV"

if [[ $1 == *.py ]]
then
    echo "python ${@:1}"
    python -u "${@:1}"
else
    echo "bash ${@:1}"
    bash "${@:1}"
fi

echo FINISHED AT $(date)
