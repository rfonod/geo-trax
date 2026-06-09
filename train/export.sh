#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)
#
# Description:
# This script exports all *.pt files in the specified directory and its 
# subdirectories to the desired format (onnx or engine).
#
# Arguments:
# 1. <input_path> - The directory containing *.pt files.
# 2. <format> - The desired export format (onnx or engine).
# 3. [-o] - Optional flag to overwrite existing exported files.
# 4. [-c <config_file>] - Optional path to a custom configuration file.
#
# Usage:
# ./train/export.sh <input_path> <format> [-o] [-c <config_file>]
#
# Examples:
# ./train/export.sh models onnx
# ./train/export.sh models onnx -o -c cfg/ultralytics/custom.yaml
#
# Notes:
# - Exported files will be saved in the same directory as the input files.
# - Ensure the script has execute permissions: chmod +x export.sh.
# - ONNX export requires: pip install -e ".[export]"
# - TensorRT (engine) export requires a CUDA-capable GPU and a separate
#   TensorRT installation (not pip-installable; install via NVIDIA SDK or
#   https://docs.nvidia.com/deeplearning/tensorrt/install-guide).
# - The default configuration file is cfg/ultralytics/default.yaml.

#-------------------------------------------------------------------------------
# Default values
#-------------------------------------------------------------------------------
batch_size=1                          # export batch size
overwrite=false                       # overwrite existing exported files
cfg_file="cfg/ultralytics/default.yaml"  # Ultralytics configuration file

#-------------------------------------------------------------------------------
# Validate positional arguments
#-------------------------------------------------------------------------------
if [ -z "$1" ]; then
    echo "Please provide an input path."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please provide the export format as the 2nd argument."
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "Input directory does not exist: $1"
    exit 1
fi

if [ "$2" == "onnx" ]; then
    format="onnx"
    device=""
elif [ "$2" == "engine" ]; then
    format="engine"
    device="device=0"
else
    echo "Invalid format: '$2'. Please select from [onnx, engine]."
    exit 1
fi

search_path="$1"
shift 2

#-------------------------------------------------------------------------------
# Parse optional flags
#-------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o)
            overwrite=true
            shift ;;
        -c)
            if [[ -n $2 ]]; then
                cfg_file="$2"
                shift 2
            else
                echo "Missing argument for -c. Expected a path to a configuration file."
                exit 1
            fi ;;
        *)
            echo "Unknown flag: $1"
            exit 1 ;;
    esac
done

#-------------------------------------------------------------------------------
# Export
#-------------------------------------------------------------------------------

# Remove trailing slash from the search path if present
search_path=$(echo "$search_path" | sed 's#/$##')

# Find all *.pt files recursively and export each one
find "$search_path" -type f -name "*.pt" |
while IFS= read -r file; do

    # Skip already-exported files unless overwrite is requested
    parent_dir=$(dirname "${file}")
    exported_file="${parent_dir}/$(basename "${file%.*}").${format}"
    if [ -f "$exported_file" ] && [ "$overwrite" = false ]; then
        echo -e "$file \033[0;33mis already exported to .${format} and overwrite is disabled.\033[0m"
        continue
    fi

    # Run the export
    echo -e "\033[1;32mExporting:\033[0m $file"
    cmd="yolo cfg=${cfg_file} mode=export model=${file} format=${format} batch=${batch_size} ${device}"
    echo -e "$cmd\n"
    $cmd
done