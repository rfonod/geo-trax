# Training Scripts

This folder contains Bash scripts for training YOLOv8 object detection models and exporting trained weights to deployment formats, using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. A SLURM wrapper is also provided for running jobs on HPC clusters.

## Overview

| Script | Description |
|--------|-------------|
| [`train.sh`](train.sh) | Main training script with flexible CLI flags for configuring and launching YOLOv8 training runs |
| [`export.sh`](export.sh) | Recursively exports `.pt` model files to ONNX or TensorRT (Engine) format |
| [`wrapper.sh`](wrapper.sh) | SLURM job submission wrapper for HPC clusters |

## Prerequisites

First set up a Python environment (venv, conda, or uv) and install geo-trax from source — see the [Installation](../README.md#installation) section of the main README. The core package covers training and Comet ML logging:

```bash
python -m pip install -e .   # pip
# uv pip install -e .        # uv
# poetry install             # Poetry
```

For model export, additional dependencies are required depending on the format:

- **ONNX** (`onnx` format): install the `export` extras:
  ```bash
  python -m pip install -e '.[export]'
  ```
- **TensorRT** (`engine` format): requires a CUDA-capable GPU and a separate TensorRT installation (not pip-installable). See the [TensorRT install guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide).

---

## `train.sh` — YOLOv8 Training

Trains a YOLOv8 model and subsequently runs multi-split validation. Outputs are organized under the `projects/` directory (created automatically if it does not exist).

### Usage

```bash
./train/train.sh [OPTIONS]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --merger-num <int>` | Merged dataset number (e.g., `6` → `merger6`) | `1` |
| `-e, --exp-num <int>` | Experiment number (e.g., `2` → `exp2`) | `1` |
| `-p, --p2` | Enable P2 neck model variant (`yolov8?-p2`) | off |
| `-rt, --rtdetr` | Use RT-DETR architecture instead of YOLOv8 | off |
| `-f, --fine-tune <int>` | Fine-tune from a prior merger's weights (0 = disabled) | `0` |
| `-fe, --fine-tune-exp <int>` | Experiment number of the fine-tune source | `1` |
| `-g, --gpus <int>` | Number of GPUs to use | `1` |
| `-r, --resume` | Resume training from the last checkpoint | off |
| `-s, --model-scale <scale>` | Model scale: `n`, `s`, `m`, `l`, `x` | `s` |
| `-d, --debug-mode` | Print commands without executing (dry run) | off |

> **Note:** `-p` and `-rt` are mutually exclusive.

### Examples

```bash
# Train merger6, experiment 2, medium-scale model on 2 GPUs
./train/train.sh -m 6 -e 2 -g 2 -s m

# Resume a previous training run
./train/train.sh -m 6 -e 2 -r

# Fine-tune from merger5 exp1, continuing as merger6 exp3
./train/train.sh -m 6 -e 3 -f 5 -fe 1

# Dry run: print all commands without executing
./train/train.sh -m 6 -e 2 -d
```

### Project directory structure

The script expects the following directory/file layout under `projects/` for each run. The run directory is created automatically before training starts:

```
projects/
└── merger<N>/
    └── <model>/<expM>/
        ├── cfg.yaml      # Ultralytics training hyperparameters
        ├── data.yaml     # Dataset paths configuration
        └── train/
            └── weights/
                ├── best.pt
                └── last.pt
```

You must create `cfg.yaml` (Ultralytics training hyperparameters) and `data.yaml` (dataset paths) before running. For `cfg.yaml`, start from the [Ultralytics default config](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml), or extract the `ultralytics:` section from a geo-trax pipeline config into a flat YAML (see [Export configuration](#export-configuration) for the one-liner).

---

## `export.sh` — Model Export

Recursively searches a directory for `.pt` files and exports them to the specified format.

### Usage

```bash
./train/export.sh <input_path> <format> [-o] [-c <config_file>]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<input_path>` | Directory to search recursively for `.pt` files |
| `<format>` | Export format: `onnx` or `engine` (TensorRT) |
| `-o` | Overwrite already-exported files |
| `-c <config_file>` | Optional Ultralytics config (a flat YAML). Omit it to use the Ultralytics built-in export defaults |

> **Note:** TensorRT (`engine`) export requires a CUDA-capable GPU and a separate TensorRT installation (not pip-installable). ONNX export has no GPU requirement. Additional packages (`onnx`, `onnxslim`, `onnxruntime`) are needed for ONNX export and are included in the `export` extras: `python -m pip install -e ".[export]"`.

#### Export configuration

`export.sh` calls Ultralytics' `yolo export`. Without `-c`, the **Ultralytics built-in export defaults** are used. Those default to `imgsz=640`, whereas geo-trax models are trained at `imgsz=1920` — so to export them faithfully, pass an Ultralytics config via `-c`. There are two easy ways to obtain one:

1. **Download the Ultralytics default** and edit it: [`ultralytics/cfg/default.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).
2. **Extract the `ultralytics:` section** from a geo-trax pipeline config into a flat YAML (it already carries `imgsz: 1920` and the geo-trax-tuned settings). `geotrax config show default` prints the pipeline config; to pull out just the detection section:
   ```bash
   python -c "import yaml; yaml.safe_dump(yaml.safe_load(open('geotrax/cfg/default.yaml'))['ultralytics'], open('ultralytics_export.yaml','w'), sort_keys=False)"
   ```

### Examples

```bash
# Export all .pt files in models/ to ONNX (Ultralytics built-in defaults)
./train/export.sh models onnx

# Export to TensorRT, overwriting existing engine files
./train/export.sh models engine -o

# Export to ONNX using a config extracted from the pipeline config (preserves imgsz=1920)
./train/export.sh models onnx -c ultralytics_export.yaml

# Export to ONNX, overwriting, with a custom config
./train/export.sh models onnx -o -c ultralytics_export.yaml
```

---

## `wrapper.sh` — SLURM Job Wrapper

A SLURM batch script that activates your Python environment and dispatches a training or inference script on an HPC cluster. It automatically detects whether the first argument is a Python (`.py`) or Bash (`.sh`) script and invokes the appropriate interpreter.

### Setup

Before using `wrapper.sh`, edit the cluster- and environment-specific lines near the top of the file:

```bash
#SBATCH --chdir /path/to/geo-trax

# Activate your Python environment (venv is the default; conda shown as a commented alternative):
source .venv/bin/activate
# eval "$(conda shell.bash hook)"
# conda activate geo-trax
```

Also adjust the resource directives (`--cpus-per-task`, `--mem`, `--time`, `--gres`, `--partition`, `--qos`) to match your cluster's policies.

### Usage

```bash
sbatch train/wrapper.sh <script> [ARGS...]
```

### Examples

```bash
# Submit a training job (merger8, experiment 1, small-scale model)
sbatch train/wrapper.sh train/train.sh -m 8 -e 1 -s s

# Submit a detection/tracking run (geotrax CLI; requires geo-trax installed in the active environment)
sbatch train/wrapper.sh geotrax extract path/to/video.mp4 --cfg geotrax/cfg/default.yaml
```

---

## Comet ML Experiment Logging

The training script integrates with [Comet ML](https://www.comet.com/) for experiment tracking (metrics, hyperparameters, model checkpoints, confusion matrices, and more). This is optional but recommended.

### Setup

1. **Install Comet ML** (already included in the `dev` extras):
   ```bash
   python -m pip install comet_ml
   ```

2. **Set your Comet API key** as an environment variable:
   ```bash
   export COMET_API_KEY=<your_api_key>
   ```
   Alternatively, store it in a `.comet.config` file in your home or project directory (see the [Comet authentication docs](https://www.comet.com/docs/v2/guides/comet-ml-authentication/)). The `.comet.config` file is listed in `.gitignore` and will **not** be committed.

3. **Comet logging is enabled automatically** by Ultralytics when `comet_ml` is installed. The training script sets `COMET_DISABLE_AUTO_LOGGING=1` to prevent duplicate logging from Comet's own auto-hook (Ultralytics handles the integration directly).

For full configuration options and environment variables (e.g., `COMET_MODE=offline` for air-gapped clusters), refer to the [Ultralytics Comet integration guide](https://docs.ultralytics.com/integrations/comet/).
