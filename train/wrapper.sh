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
#   sbatch train/wrapper.sh geotrax extract path/to/video.mp4 --cfg geotrax/cfg/default.yaml
#
# Notes:
#   - Update --chdir to the absolute path of your geo-trax project root.
#   - Edit the "Activate your Python environment" block below to match your setup
#     (venv, conda, or uv). venv is the default; conda is shown as a commented alternative.
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
# Activate your Python environment (edit to match your setup: venv, conda, or uv)
#-------------------------------------------------------------------------------
echo -e "\nActivating Python environment..."
# Option A (default): venv / uv — activate the environment created with `python -m venv` or `uv venv`
source .venv/bin/activate
# Option B: conda — uncomment and set your environment name
# eval "$(conda shell.bash hook)"
# conda activate geo-trax
echo -e "Using Python: $(which python)"

if [[ $1 == *.py ]]
then
    echo "python ${@:1}"
    python -u "${@:1}"
elif [[ $1 == geotrax ]]
then
    echo "python -m geotrax ${@:2}"
    python -u -m geotrax "${@:2}"
else
    echo "bash ${@:1}"
    bash "${@:1}"
fi

echo FINISHED AT $(date)
