#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)
#
# Description:
# This script trains a YOLOv8 (or RT-DETR) object detection model using the
# Ultralytics framework. After training completes, it runs validation on the
# test split (and optionally as single-class). All outputs are saved under
# the projects/ directory, which is created automatically if it does not exist.
#
# Usage:
# ./train/train.sh [OPTIONS]
#
# Options:
#   -m, --merger-num <int>      Dataset number to train on (e.g., 6 -> merger6) [default: 1]
#   -e, --exp-num <int>         Experiment number (e.g., 2 -> exp2)            [default: 1]
#   -p, --p2                    Use P2 neck model variant (yolov8?-p2)
#   -rt, --rtdetr               Use RT-DETR architecture instead of YOLOv8
#   -f, --fine-tune <int>       Fine-tune from a prior dataset's best weights   [default: 0 (off)]
#   -fe, --fine-tune-exp <int>  Experiment number of the fine-tune source       [default: 1]
#   -g, --gpus <int>            Number of GPUs to use                           [default: 1]
#   -r, --resume                Resume training from the last checkpoint
#   -s, --model-scale <scale>   Model scale: n, s, m, l, x                     [default: s]
#   -d, --debug-mode            Dry run: print commands without executing
#
# Notes:
#   - Flags -p and -rt are mutually exclusive.
#   - cfg.yaml and data.yaml must exist in projects/<dataset>/<model>/<exp>/
#     before running (they define training hyperparameters and dataset paths).
#   - Set COMET_API_KEY in your environment to enable experiment tracking.
#
# Examples:
#   ./train/train.sh -m 6 -e 2 -g 2 -s m     # train merger6/exp2 on 2 GPUs, medium scale
#   ./train/train.sh -m 6 -e 2 -r            # resume merger6/exp2
#   ./train/train.sh -m 6 -e 3 -f 5 -fe 1    # fine-tune from merger5/exp1 as merger6/exp3
#   ./train/train.sh -m 6 -e 2 -d            # dry run

echo "Bash version: $BASH_VERSION"

#-------------------------------------------------------------------------------
# Default values
#-------------------------------------------------------------------------------
arg_merger_num=1     # dataset number (1 → merger1, 2 → merger2, ...)
arg_exp_num=1        # experiment number (1 → exp1, 2 → exp2, ...)
arg_p2=0             # use P2 neck model variant (0 - off, 1 - on)
arg_rtdetr=0         # use RT-DETR architecture (0 - off, 1 - on)
arg_fine_tune=0      # fine-tune source dataset number (0 - disabled)
arg_fine_tune_exp=1  # fine-tune source experiment number
arg_gpus=1           # number of GPUs to use
arg_resume=0         # resume from last checkpoint (0 - off, 1 - on)
arg_model_scale="s"  # model scale: n, s, m, l, x  (RT-DETR supports only l, x)
arg_debug_mode=0     # dry run: print commands without executing (0 - off, 1 - on)

# GPU resource monitoring (only active when check_resources=1)
devices={0,1}        # brace-expansion pattern for GPU indices (e.g., {0,1} → GPUs 0 and 1)
check_resources=0    # poll for a free GPU before launching (0 - off, 1 - on)
                     # N.B.: when enabled, overwrites the device derived from arg_gpus

#-------------------------------------------------------------------------------
# Argument parsing
#-------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--merger-num)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                arg_merger_num=$2
                shift 2
            else
                echo "Invalid argument for -m|--merger-num. Expected a positive integer."
                exit 1
            fi ;;
        -e|--exp-num)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                arg_exp_num=$2
                shift 2
            else
                echo "Invalid argument for -e|--exp-num. Expected a positive integer."
                exit 1
            fi ;;
        -p|--p2)
            arg_p2=1
            shift ;;
        -rt|--rtdetr)
            arg_rtdetr=1
            shift ;;            
        -f|--fine-tune)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                arg_fine_tune=$2
                shift 2
            else
                echo "Invalid argument for -f|--fine-tune. Expected a positive integer."
                exit 1
            fi ;;
        -fe|--fine-tune-exp)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                arg_fine_tune_exp=$2
                shift 2
            else
                echo "Invalid argument for -fe|--fine-tune-exp. Expected a positive integer."
                exit 1
            fi ;;
        -g|--gpus)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                arg_gpus=$2
                shift 2
            else
                echo "Invalid argument for -g|--gpus. Expected a positive integer."
                exit 1
            fi ;;
        -r|--resume)
            arg_resume=1
            shift ;;
        -d|--debug-mode)
            arg_debug_mode=1
            shift ;;
        -s|--model-scale)
            if [[ -n $2 && $2 =~ ^(n|s|m|l|x)$ ]]; then
                arg_model_scale=$2
                shift 2
            else
                echo "Invalid argument for -s|--model-scale. Available options: n, s, m, l, x."
                exit 1
            fi ;;
        *)
            echo "Invalid flag: $1"
            exit 1
            ;;
    esac
done

if [ "$arg_p2" == "1" ] && [ "$arg_rtdetr" == "1" ]; then
    echo "Invalid flags: -p and -rt cannot be set at the same time."
    exit 1
fi

#-------------------------------------------------------------------------------
# Print the values of the variables
#-------------------------------------------------------------------------------
echo -e "\nThe following variables are set:"
echo "arg_merger_num: $arg_merger_num"
echo "arg_exp_num: $arg_exp_num"
echo "arg_p2: $arg_p2"
echo "arg_rtdetr: $arg_rtdetr"
echo "arg_fine_tune: $arg_fine_tune"
echo "arg_fine_tune_exp: $arg_fine_tune_exp"
echo "arg_gpus: $arg_gpus"
echo "arg_resume: $arg_resume"
echo "arg_model_scale: $arg_model_scale"
echo "arg_debug_mode: $arg_debug_mode"

#-------------------------------------------------------------------------------
# Set the variables
#-------------------------------------------------------------------------------
dataset=merger$arg_merger_num
experiment=exp$arg_exp_num  # exp1, exp2, ...

if [ "$arg_p2" == "1" ]; then
    model=yolov8$arg_model_scale-p2
elif [ "$arg_rtdetr" == "1" ]; then
    model=yolov8${arg_model_scale}-rtdetr
else
    model=yolov8$arg_model_scale
fi

if [ "$arg_fine_tune" == "0" ]; then
    if [ "$arg_rtdetr" == "1" ]; then
        weights=${model}.pt
    else
        weights=yolov8$arg_model_scale.pt # yolov8?-p2.pt are not available
    fi
else
    weights=projects/merger${arg_fine_tune}/${model}/exp${arg_fine_tune_exp}/train/weights/best.pt
fi

weights_resume=projects/${dataset}/${model}/${experiment}/train/weights/last.pt
weights_best=projects/${dataset}/${model}/${experiment}/train/weights/best.pt

if [ $check_resources == "0" ]; then
    if [ $arg_gpus == "1" ]; then
        device=0
    else
        device="0,$(seq -s, 1 $(($arg_gpus-1)))"
        # check if the last character of device is a comma, if yes, remove it
        if [ "${device: -1}" == "," ]; then
            device=${device::-1}
        fi
    fi
fi

#-------------------------------------------------------------------------------
# Derived variables
#-------------------------------------------------------------------------------
run_name=${model}/${experiment}
model_yaml="${model}.yaml"
cfg_file="projects/${dataset}/${run_name}/cfg.yaml"
data_file="projects/${dataset}/${run_name}/data.yaml"

export COMET_DISABLE_AUTO_LOGGING=1

#-------------------------------------------------------------------------------
# Test resources
#-------------------------------------------------------------------------------

resources () { 
    for device in $( eval echo $devices )
    do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $device | grep -Eo [0-9]+) 
        free_gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv -i $device | grep -Eo [0-9]+)
        
        if [ $free_mem -lt 10000 ] || [ $free_gpu -gt 10 ]; then
            condition=true
        else
            condition=false
        fi
        echo "GPU #${device} (GPU MEM = ${free_mem}MiB | GPU UTIL = ${free_gpu}%) is NOT available -> ${condition}"
        
        if [ $condition == false ]; then
            break
        fi
    done
}

if [ $check_resources == "1" ]; then
    resources
    while [ $condition == true ]
    do
        echo -e "Waiting for resources..."
        resources
        sleep 60
    done
fi

#-------------------------------------------------------------------------------
# Define commands
#-------------------------------------------------------------------------------

echo -e "\nRunning: projects/$dataset/$run_name\n"

# 1) train command
if [ "$arg_resume" == "1" ]; then
    echo -e "Resuming training...\n"
    cmd_train="yolo cfg=${cfg_file} mode=train model=${weights_resume} data=${data_file} device=${device} project=projects/${dataset} name=${run_name}/train resume"
else
    echo -e "Starting training...\n"
    if [ "$arg_fine_tune" == "0" ]; then
        cmd_train="yolo cfg=${cfg_file} mode=train model=${model_yaml} pretrained=${weights} data=${data_file} device=${device} project=projects/${dataset} name=${run_name}/train"
    else
        cmd_train="yolo cfg=${cfg_file} mode=train model=${weights} data=${data_file} device=${device} project=projects/${dataset} name=${run_name}/train"
    fi
fi  

# 2) validation command for test set
cmd_test="yolo cfg=${cfg_file} mode=val split=test model=${weights_best} data=${data_file} device=${device::1} project=projects/${dataset} name=${run_name}/test"

# 3) validation command for test set (eval as single-class)
cmd_test_sc="yolo cfg=${cfg_file} mode=val split=test model=${weights_best} data=${data_file} device=${device::1} project=projects/${dataset} name=${run_name}/test-sc single_cls"

#-------------------------------------------------------------------------------
# Run commands
#-------------------------------------------------------------------------------

if [ "$arg_debug_mode" == "1" ]; then
    echo -e "$cmd_train\n"
    echo -e "$cmd_test\n"
    echo -e "$cmd_test_sc\n"
else
    mkdir -p "projects/${dataset}/${run_name}"
    echo -e "$cmd_train\n" > projects/${dataset}/${run_name}/commands.txt
    $cmd_train
    echo -e "$cmd_test\n" | tee -a projects/${dataset}/${run_name}/commands.txt
    $cmd_test
    echo -e "$cmd_test_sc\n" | tee -a projects/${dataset}/${run_name}/commands.txt
    $cmd_test_sc
fi
