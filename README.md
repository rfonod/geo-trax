# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![PyPI - Version](https://img.shields.io/pypi/v/geo-trax)](https://pypi.org/project/geo-trax/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/geo-trax?label=total%20downloads)](https://pepy.tech/project/geo-trax) [![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/geo-trax?color=%234c1)](https://pypi.org/project/geo-trax/) [![CI](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/issues) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![Hugging Face](https://img.shields.io/badge/🤗%20Model-rfonod%2Fgeo--trax-yellow)](https://huggingface.co/rfonod/geo-trax) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/rfonod/geo-trax) [![Project Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax) [![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk)

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is a comprehensive pipeline that extracts high-accuracy, georeferenced vehicle trajectories from high-altitude drone imagery. Built for quasi-stationary aerial monitoring of urban traffic, it turns raw bird's-eye view (BEV) drone footage into precise, real-world vehicle trajectories. The framework combines YOLO detection, multi-object tracking, and video stabilization with a robust orthophoto-based georeferencing stage, producing geo-coordinated, lane-resolved trajectories that are spatially and temporally consistent and ready for large-scale traffic analysis and simulation. It is optimized for urban intersections and arterial corridors, where high-fidelity, vehicle-level insights drive intelligent transportation systems and digital twin applications.

![Geo-trax Output Visualization](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization.webp)

🎬 An accelerated preview of Geo-trax's capabilities. Watch the full ~4 min 4K demo on [YouTube](https://youtu.be/gOGivL9FFLk).

> [!TIP]
> **Just want to see it work?** Try the [interactive demo on 🤗 Hugging Face Spaces](https://huggingface.co/spaces/rfonod/geo-trax): run the vehicle detector on your own aerial image or short clip right in the browser, no install required.

### Why Geo-trax

- 🛰️ **Real-world output**: georeferenced, lane-resolved trajectories (WGS84 + local CRS) with per-vehicle speed, acceleration, and estimated dimensions, straight from raw BEV drone video.
- 🎯 **Accurate detection**: [YOLOv8s vehicle detector](#detection-model) reaching **0.951 mAP@50**, trained on more than 19,000 annotated aerial images.
- 🚗 **Flexible tracking**: four vehicle classes and [six selectable multi-object trackers](#tracking) (BoT-SORT, ByteTrack, OC-SORT, and more).
- ⚖️ **Drone-motion robust**: homography-based [stabilization](#stabilization) ([Stabilo](https://github.com/rfonod/stabilo)) plus orthophoto image [registration](#georeferencing) for consistent, cross-flight coordinates; both optionally CUDA-accelerated.
- 📊 **Proven at scale**: powered the [Songdo Traffic](https://doi.org/10.5281/zenodo.13828383) dataset (roughly **700,000 trajectories** across **20 intersections**, fleet of **10 drones**; see [Real-World Deployment](#real-world-deployment-the-songdo-experiment)).
- ⚙️ **One command, one config**: `geotrax batch` runs the whole pipeline; a single YAML drives every stage, with [four tuned presets](#configuration) included.

## Pipeline

![Geo-trax pipeline diagram: raw drone video → detection → tracking → stabilization → georeferencing → dataset](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_pipeline.svg)

🔍 The core pipeline (solid box) produces stabilized, pixel-coordinate vehicle trajectories. Optional extensions add georeferencing via orthophoto image registration, vision dataset creation through frame (pre-)annotation for custom detector fine-tuning, and visualization, analysis, and probe vehicle validation tools, all applicable to both pixel-coordinate and georeferenced outputs.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install geo-trax
```

Python 3.11 to 3.13. Also works with [uv](https://docs.astral.sh/uv/) (`uv pip install geo-trax`) and [conda](https://www.anaconda.com/docs/getting-started/miniconda/install). For development:

```bash
git clone --depth 1 https://github.com/rfonod/geo-trax.git
cd geo-trax && python -m pip install -e '.[dev]'
```

> [!NOTE]
> The default model auto-downloads from [🤗 Hugging Face](https://huggingface.co/rfonod/geo-trax) on first use (cached in `~/.cache/huggingface/hub`, overridable via `HF_HOME`). To use your own weights, set `--model` or `extraction.model` in the config to a local `.pt` path or `hf://<org>/<repo>/<path/to/file>.pt`.

<details>
<summary><b>Alternative Environments & Advanced Dev Install</b></summary>

**Create and activate a virtual environment** (any of the following):

```bash
# venv (standard library)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# uv (fastest drop-in for venv + pip)
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Miniconda
conda create -n geo-trax python=3.11 -y
conda activate geo-trax
```

**Install from PyPI** (runtime use). This installs the `geotrax` command-line interface together with the bundled configuration tree (`geotrax/cfg/`):

```bash
python -m pip install geo-trax    # pip
uv pip install geo-trax           # uv (faster)
```

**Install from local source** (recommended for development or model training). Clone (or fork) the repository, then install in editable mode (`-e`), which reflects code changes without reinstalling:

```bash
git clone https://github.com/rfonod/geo-trax.git   # add --depth 1 for the latest snapshot only
cd geo-trax
python -m pip install -e .         # pip
# uv pip install -e .              # uv (faster; requires the uv venv above)
# poetry install                   # Poetry (auto-manages its own virtualenv; skip the venv step)
```

**Optional dependency groups** (development/testing tools, ONNX export, SAHI sliced inference):

```bash
python -m pip install -e '.[dev]'      # development + test tooling
python -m pip install -e '.[export]'   # ONNX export dependencies
python -m pip install -e '.[sahi]'     # SAHI sliced inference for small-object detection (--sahi)
# uv pip install -e '.[dev]'           # uv equivalents
# poetry install --extras dev          # Poetry equivalents
# poetry install --extras export
# poetry install --extras sahi
```

Installed geo-trax from PyPI rather than a checkout? Use the non-editable form instead: `python -m pip install 'geo-trax[dev]'` (likewise `'geo-trax[export]'`, `'geo-trax[sahi]'`).

**Optional CUDA for image matching.** The stabilization (`--stab-gpu`) and georeferencing (`--geo-gpu`) steps can be CUDA-accelerated on top of a source-built OpenCV. Installing into such an environment needs care so the CPU OpenCV wheels do not overwrite your build; see [GPU acceleration](#gpu-acceleration). (Object *detection* already uses CUDA automatically when available.)

</details>

## Quick Start

`data/U_video_cut.mp4` is a 5-second sample clip included for immediate testing. See [data/README.md](data/README.md) for matching orthophotos.

```bash
# Pixel-coordinate trajectories (no orthophoto required)
geotrax batch data/U_video_cut.mp4 --no-geo

# Full pipeline: extract, georeference, and analyze (orthophotos required; see data/README.md)
geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -mf data/master_frames --show-lanes

# Scale up: process a whole project tree, then merge multi-drone results into one dataset
geotrax batch path/to/PROCESSED/
geotrax aggregate path/to/PROCESSED/
```

Run `geotrax -h` or `geotrax batch -h` for all options. The scale-up commands above run flag-free with the [recommended project structure](#real-world-deployment-the-songdo-experiment); any other layout works with explicit path flags.

<details>
<summary><b>📋 Full Feature Overview</b></summary>

- **Detection**: YOLOv8s on aerial BEV imagery; detects car (incl. vans), bus, truck, and motorcycle; optional [SAHI](https://github.com/obss/sahi) sliced inference for improved small-object recall (`--sahi`).
- **Tracking**: six multi-object trackers (BoT-SORT default); see [Tracking](#tracking) for a comparison; optional per-track frame-gap interpolation.
- **Stabilization**: homography-based trajectory correction via [Stabilo](https://github.com/rfonod/stabilo) ⚖️, tuned with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯; optional CUDA acceleration (`--stab-gpu`).
- **Georeferencing**: frame-to-orthophoto registration; outputs lat/lon, local CRS, speed, acceleration, and lane assignment per vehicle; optional CUDA acceleration (`--geo-gpu`).
- **Visualization**: track overlays on original, stabilized, or static-reference video, in five rendering modes (incl. oriented bounding boxes).
- **Analysis**: trajectory maps, kinematic distributions, and class/dimension charts, per-video or aggregated across drones and sessions.
- **Scaling & tooling**: batch-processes directory trees and aggregates multi-drone data; includes standalone utilities for end-to-end data preparation, training, evaluation, and validation.

</details>

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- Comprehensive documentation in a dedicated `docs/` folder. A [`tools/README.md`](tools/README.md) index already covers the auxiliary scripts.
- Modularized, OOP-based pipeline with custom reference frame support and georeferencing leveraging Stabilo's image-matching backend.
- Per-class confidence thresholds.
- Simulation and scenario export: `geotrax export --format sumo` (and/or OpenDRIVE, CommonRoad) to turn an extracted dataset into a drop-in scenario for traffic simulators and digital twins, reusing the lane and road-section geometry already carried by the orthophoto segmentations.
- Live/online mode: ingest RTMP-style drone feeds for real-time georeferenced output, for live monitoring and incident detection rather than offline batches.
- Batch inference and multi-thread processing, plus distributed fan-out (Ray, Dask, or Slurm) so `geotrax batch` scales across a cluster for multi-drone, multi-intersection campaigns.
- Real-world map visualization (e.g., MovingPandas, contextily) and an interactive web viewer with map-based playback and filtering by class, lane, speed, and time, so datasets can be explored without writing plotting code.

</details>

<details>
<summary><b>🔗 Related Projects</b></summary>

Geo-trax integrates with and complements several specialized tools:

- **[Stabilo](https://github.com/rfonod/stabilo) ⚖️**: Python library for video and trajectory stabilization using robust homography transformations. Supports various feature detectors, RANSAC algorithms, and user-defined masks. Used as Geo-trax's core stabilization engine.

- **[Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯**: benchmarking and hyperparameter optimization framework for Stabilo. Evaluates stabilization performance through ground truth-free assessment using random perturbations. Used to fine-tune Geo-trax stabilization parameters.

- **[HBB2OBB](https://github.com/rfonod/hbb2obb) 📦**: converts horizontal bounding boxes to oriented bounding boxes using SAM segmentation models. Can enhance Geo-trax outputs when object orientation is needed for downstream analysis.

</details>

## Configuration

The entire pipeline is driven by a **single, self-contained YAML config**: one file for detection, tracking, stabilization, georeferencing, visualization, and plotting. Four presets ship with the package:

| Preset | Focus |
|--------|-------|
| [`default`](geotrax/cfg/default.yaml) | Balanced baseline |
| [`confident`](geotrax/cfg/confident.yaml) | Precision (fewer false positives) |
| [`lenient`](geotrax/cfg/lenient.yaml) | Recall (catches more vehicles) |
| [`stable`](geotrax/cfg/stable.yaml) | Stabilization quality |

```bash
geotrax batch video.mp4 -c confident   # use a bundled preset by name
geotrax batch video.mp4 -c ./my.yaml   # use a custom config file
```

<details>
<summary><b>⚙️ Inspect, copy, and customize configs</b></summary>

Manage the bundled configs with the `geotrax config` command:

```bash
geotrax config show              # list bundled presets and their location
geotrax config show default      # print a preset's full contents
geotrax config copy              # copy presets into the current directory as <name>_copy.yaml
geotrax config copy -o ~/myproj  # copy into a specific directory
```

Copy a preset, edit it, then pass it with `-c`:

```bash
geotrax config copy
# edit default_copy.yaml ...
geotrax extract video.mp4 -c default_copy.yaml
```

To switch the tracking algorithm, set `tracker.active` in the config (see [Tracking](#tracking)).

</details>

<details>
<summary><b>✏️ One-off overrides with <code>--set</code></b></summary>

Change any config value for a single run, without copying a config file. Available on every command that takes `-c`:

```bash
geotrax extract video.mp4 --set conf=0.35 --set iou=0.6     # repeat --set for several keys
geotrax extract video.mp4 --set tracker.botsort.track_buffer=45
geotrax georeference video.mp4 -orf ortho/ --set matching.downsample_ratio=0.25
geotrax batch PROCESSED/ --set stabilo.max_features=4000 --set visualization.tail_length=90
```

- **KEY** is a dotted path (`ultralytics.conf`) or any unambiguous tail of one (`conf`, `matching.gpu`). An ambiguous or misspelled key stops the run and lists the candidates rather than being silently ignored. The full dotted path is the stable form: a short name that is unique today could become ambiguous once a future release adds a config key.
- **VALUE** is read with YAML rules, exactly as in the config file, so `true`, `null`, `0.35`, and `[0, 1, 2]` all mean what they look like. A value of the wrong kind (text where a number belongs, a scalar where a list belongs) is rejected; numbers stay interchangeable.
- **Precedence** is CLI > config file. Every override is echoed to the console, so it is never silent.

Where a dedicated flag exists, prefer it: flags validate their input (`--stab-detector xfeatt` fails immediately and lists the valid detectors, while `--set` can only check the type) and they show up in `--help`. Passing both for the same key is an error, not a precedence puzzle:

```console
$ geotrax extract video.mp4 --set conf=0.1 --conf 0.5
CRITICAL - Conflicting overrides for 'ultralytics.conf':
  --set conf=0.1
  --conf 0.5
Pass only one.
```

`--set` is for experimentation; to make a change durable, copy a preset and pass it with `-c`.

</details>

## Detection Model

The default detector is **YOLOv8s** (HBB, 1920 × 1920 px, ~11 M parameters), trained on more than 19,000 annotated aerial images (~679k labeled vehicle instances) and fine-tuned on a curated, high-quality subset. It is hosted on [🤗 Hugging Face](https://huggingface.co/rfonod/geo-trax) and **downloads automatically on first use**. Results on the [Songdo Vision **v1**](https://doi.org/10.5281/zenodo.13828408) test split (1,084 images; full results in [Table 3](https://doi.org/10.1016/j.trc.2025.105205)):

| ID | Label | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| 0 | Car (incl. vans) | 0.979 | 0.981 | 0.992 | 0.835 |
| 1 | Bus | 0.952 | 0.977 | 0.988 | 0.826 |
| 2 | Truck | 0.887 | 0.916 | 0.935 | 0.722 |
| 3 | Motorcycle | 0.827 | 0.866 | 0.888 | 0.463 |
| **All** | | **0.911** | **0.935** | **0.951** | **0.711** |

> Pedestrian and bicycle classes exist in the weights but are underrepresented, unevaluated, and filtered by default. See the [model card](https://huggingface.co/rfonod/geo-trax) for full details.

To use a different model, point `--model` (CLI) or `extraction.model` (config) to a local `.pt` path or `hf://<org>/<repo>/<file>.pt`; any [Ultralytics](https://github.com/ultralytics/ultralytics)-compatible model works.

### Small-Object Detection with SAHI

For footage where vehicles are near the detection floor (higher altitudes, sub-4K sensors), the extraction stage can optionally run [SAHI](https://github.com/obss/sahi) sliced inference: each frame is split into overlapping slices, the detector runs on every slice (plus one full-frame pass), and the results are merged before tracking. This substantially improves tiny-object recall at roughly 5x the per-frame detection cost with the default 1920 x 1080 slices.

```bash
geotrax extract video.mp4 --sahi     # also available on 'geotrax batch'
```

SAHI is an optional extra. If it is not installed, the run stops immediately with the exact install command for your setup.

Slice size, overlap, and merge settings live in `cfg -> extraction -> sahi` (run `geotrax config copy` to edit them). SAHI mode honors the `ultralytics` config keys `conf`, `device`, `imgsz` (applied per slice), and `classes`; the NMS-related keys (`iou`, `max_det`, `agnostic_nms`, etc.) are superseded by the SAHI merge settings. It supports YOLO models only and cannot be combined with the `tracktrack` tracker or with ReID model `auto` (both need a live Ultralytics predictor).

### Custom Model Training

Training and export scripts for custom YOLO detectors live in `train/`, with a SLURM wrapper for HPC clusters. See [train/README.md](train/README.md).

## Tracking

Six multi-object trackers ship with [Ultralytics](https://github.com/ultralytics/ultralytics) `>=8.4.63`. Selection is config-driven: set `tracker.active`, no code changes needed. Default: **BoT-SORT**.

| Tracker | `tracker.active` | ReID | GMC¹ | Pros | Cons |
|---------|------------------|:----:|:----:|------|------|
| **BoT-SORT** (default) | `botsort` | opt | ✅ | Strong accuracy; motion + optional appearance | Slower; ReID adds compute |
| **ByteTrack** | `bytetrack` | ❌ | ❌ | Fastest; two-stage association | More ID switches under occlusion |
| **OC-SORT** | `ocsort` | ❌ | ❌ | Robust to non-linear motion; lightweight | Weaker on long occlusions |
| **Deep OC-SORT** | `deepocsort` | opt | opt | OC-SORT + appearance; dense scenes | Heaviest variant with ReID |
| **FastTracker** | `fasttrack` | ❌ | ❌ | Occlusion-aware ByteTrack variant | Newer; several knobs to tune |
| **TrackTrack** | `tracktrack` | opt | ✅ | Multi-cue cost; best ID retention | Most parameters; highest compute |

¹ GMC (in-tracker camera-motion compensation) runs during tracking and is independent of Stabilo's post-hoc trajectory stabilization stage.

> 💡 Run `geotrax config show default` to print the full `tracker:` block, with every parameter for all six trackers documented inline. Run `geotrax config copy` to get an editable local copy. For a head-to-head comparison on your own data, see [`tools/compare_tracking.py`](tools/compare_tracking.py).

## Stabilization

Quasi-stationary drone footage is never perfectly static: wind, GPS drift, and gimbal corrections shift the frame throughout a flight. Geo-trax compensates by registering every frame back to a fixed reference frame with a homography, via [Stabilo](https://github.com/rfonod/stabilo) ⚖️. Tracks are written in both raw and stabilized image coordinates, and the per-frame 3 × 3 transform is saved alongside them.

Registration quality depends on the feature detector. The default is `orb`, tuned for high-altitude BEV footage with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯 and used for every published result:

| Detector | Description |
|---|---|
| `orb` (default) | fast binary descriptor; the only CUDA-accelerated option |
| `sift` | classic scale-invariant descriptor |
| `rsift` | RootSIFT, a normalized SIFT variant with better matching accuracy |
| `brisk` | binary, scale- and rotation-invariant |
| `kaze` | nonlinear scale space, slower but accurate |
| `akaze` | accelerated KAZE, a good speed/accuracy compromise |

All six are classical OpenCV detectors, rotation invariant, and need no extra downloads:

```bash
geotrax extract video.mp4 --stab-detector akaze
```

Masking, RANSAC, and downsampling parameters live under `cfg -> stabilo`. To skip stabilization entirely, set `extraction.stabilize: false`.

<details>
<summary><b>Learning-based detectors</b></summary>

Stabilo 1.4.0 also exposes five [kornia](https://kornia.readthedocs.io/) models: `xfeat`, `disk`, `dedode`, `keynet`, and the detector-free `loftr`, plus the learned `lightglue` matcher (which pairs only with `disk`/`dedode`/`keynet`). They are available for experimentation on both the stabilization and georeferencing stages, but they are **not** used by any geo-trax default and were not used to produce any published result. On this workload the classical detectors are faster, more predictable, and easier to reproduce.

Two properties make them awkward here:

- **Rotation.** `xfeat`, `disk`, `dedode`, and `loftr` are *upright* models: matching collapses once the two images differ by more than roughly 30°. That is tolerable for stabilization, where consecutive frames are nearly aligned, but a poor fit for georeferencing, where a drone frame is registered against a north-up orthophoto and arbitrary rotation is the norm. Only `keynet` estimates keypoint orientation.
- **Memory.** They are far hungrier than classical detectors, and cost grows with the *processed* frame size. **This can exhaust system memory and hang the machine.** Pick `downsample_ratio` so that `input_pixels × downsample_ratio²` stays under the model's guideline:

  | Detector | Guideline | Max ratio at 4K (8.3 MP) | Max ratio at 8000 × 8000 ortho (64 MP) |
  |---|---|---|---|
  | `xfeat` | 2.0 MP | `0.49` | `0.17` |
  | `disk`, `dedode`, `keynet` | 0.5 MP | `0.24` | `0.08` |
  | `loftr` | 0.3 MP (cost is **quadratic** in pixels) | `0.19` | `0.06` |

  Stabilo warns when the processed frame exceeds its guideline but does not cap it, so heed the warning. geo-trax does cap `max_features` on the georeferencing path, since these models consume it as `top_k` and the RootSIFT-sized default of 250000 would be fatal.

**Acceleration.** These models run on PyTorch, so they are GPU-accelerated straight from the CLI with `--stab-device` / `--geo-device` (`auto` picks cuda > mps > cpu, and is the default). Nothing extra is needed: unlike the OpenCV CUDA path in [GPU acceleration](#gpu-acceleration), which requires a source-built OpenCV and covers only ORB, this works on stock wheels, accelerates every learned detector and the `lightglue` matcher, and supports `mps` on Apple silicon. A GPU with dedicated VRAM is also the safer choice memory-wise, since it does not compete with system RAM. The two paths are mutually exclusive; Stabilo rejects `--stab-gpu`/`--geo-gpu` combined with a learned detector.

Weights download on first use and are cached by torch (`~/.cache/torch/hub/checkpoints`; set `TORCH_HOME` to relocate), then reused offline. Set the ratio well below the maximum before the first run:

```bash
geotrax extract video.mp4 --stab-detector xfeat --stab-device cuda -c my_config.yaml
#   my_config.yaml: stabilo.downsample_ratio: 0.25
```

</details>

## Georeferencing

The georeferencing stage turns stabilized pixel trajectories into real-world ones by registering the video's reference frame against a north-up **orthophoto** of the same location. Output is a per-vehicle CSV with WGS84 coordinates, a local projected CRS, speed, acceleration, and lane / road-section assignment.

This reuses the image-matching machinery from stabilization, but defaults to **`rsift`** (RootSIFT) instead of `orb`. Registering a drone frame against an orthophoto is a cross-domain match, often across years, seasons, and lighting conditions, and is considerably harder than matching consecutive video frames:

```bash
geotrax georeference video.mp4 -orf data/orthophotos --geo-detector sift
```

Orthophotos are required; lane/road-section segmentations and per-location master frames are optional and improve lane assignment and robustness (see [Usage](#usage) and [data/README.md](data/README.md) for how to obtain them). Matching parameters live under `cfg -> georef -> matching`, kinematic filtering under `cfg -> georef -> filtering`.

> The [learning-based detectors](#stabilization) are a poor fit for this stage in particular: the orthophoto is north-up while the drone frame can be at any heading, and every learned model except `keynet` is upright-only.

## GPU acceleration

Object **detection** already runs on CUDA automatically whenever a compatible GPU and PyTorch build are present (via the `ultralytics.device` config key, auto by default). The **stabilization** (`--stab-gpu`) and **georeferencing** (`--geo-gpu`) image-matching steps can *optionally* be CUDA-accelerated too, through [Stabilo](https://github.com/rfonod/stabilo). This needs a CUDA-enabled OpenCV build and is Linux/Windows only. Stabilo accelerates the **ORB** detector only, so `--geo-gpu` additionally requires `--geo-detector orb`; there is no CPU fallback, so requesting GPU without a working CUDA device raises an error.

> This section covers the **OpenCV CUDA** path, which applies to the classical detectors. The learning-based detectors use a separate torch device (`--stab-device` / `--geo-device`) and are covered under [Stabilization](#stabilization); the two cannot be combined.

<details>
<summary><b>⚡ Full CUDA setup & benchmarking guide</b></summary>

Throughout this guide, dotted names like `ultralytics.device` or `georef.matching.detector_name` are **keys in the pipeline config**, not CLI flags. To change them, copy the bundled config once and pass your copy with `-c`:

```bash
geotrax config copy                 # writes default_copy.yaml in the current directory
# edit default_copy.yaml, then pass it to any command:
geotrax batch <video> ... -c default_copy.yaml
```

### 1. Build OpenCV with CUDA

The PyPI OpenCV wheels are CPU-only, so you must build `opencv-contrib-python` from source with CUDA. Follow Stabilo's [`docs/cuda.md`](https://github.com/rfonod/stabilo/blob/main/docs/cuda.md), then confirm it works:

```bash
python -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"   # expect: <version> 1
```

> **geo-trax needs one OpenCV module beyond the minimal stabilo build: add `video` to the `BUILD_LIST`.** The minimal list in stabilo's guide omits it, but the `sparseOptFlow` global-motion-compensation (GMC) method uses `cv2.calcOpticalFlowPyrLK` from that module. It is the default for the **BoT-SORT** (active default) and **TrackTrack** trackers, and an option for **DeepOCSORT** (`tracker.<name>.gmc_method`). GMC runs during **track association on the original, pre-stabilization frames**, so Stabilo does not make it redundant. Without `video` you get a `cv2 has no attribute 'calcOpticalFlowPyrLK'` warning and GMC silently falls back to identity. If you would rather not rebuild, set `gmc_method: none` for your active tracker (step 3), but that disables camera-motion compensation in tracking; for near-nadir BEV / quasi-stationary drone footage that is often acceptable, but check your tracking quality before relying on it.

### 2. Install geo-trax without clobbering your CUDA OpenCV

`ultralytics[extra]` transitively requires `opencv-python` and `opencv-python-headless`, both CPU wheels. A plain install drops them on top of your compiled `cv2/` and disables CUDA. A transitive dependency cannot be excluded in `pyproject.toml`, so use one of these:

**A. Install, then restore your wheel (recommended).** Let pip install everything, then overwrite the CPU OpenCV by reinstalling the CUDA `opencv_contrib_python-*.whl` you built in step 1:

```bash
pip install -e .          # installs torch/cuda/ultralytics/..., plus (temporarily) CPU opencv
# reinstall the wheel produced by your CUDA OpenCV build in step 1 (path is wherever you built it):
pip install --force-reinstall --no-deps /path/to/opencv_contrib_python-*.whl
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"   # expect: 1
```

> If the CUDA check above reports `0` devices (or a `cv2`/CUDA error such as `module 'cv2' has no attribute 'cuda'`) immediately after the reinstall, re-activate the CUDA venv (`source .venv-cuda/bin/activate`, matching stabilo's `docs/cuda.md`) and run it again; if it persists, confirm `cv2.__file__` points into that venv's `site-packages` (see the troubleshooting in stabilo's `docs/cuda.md`). It should then report `1`.
>
> ⚠️ Afterwards `opencv-python` / `opencv-python-headless` stay registered but their recorded files now belong to your build. **Never** `pip uninstall opencv-python` / `opencv-python-headless`, or you will delete the shared `cv2/`.

**B. Stub the CPU wheels first (no download, cleaner metadata).** Install two metadata-only packages so pip treats the requirements as already satisfied and never fetches a CPU wheel:

```bash
CVVER=$(python -c "import cv2; print(cv2.__version__)")
for pkg in opencv-python opencv-python-headless; do
  d=$(mktemp -d)
  cat > "$d/pyproject.toml" <<EOF
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "$pkg"
version = "$CVVER"
[tool.setuptools]
py-modules = []
EOF
  pip install --no-deps "$d" && rm -rf "$d"
done
pip install -e .          # opencv is satisfied by the stubs; your CUDA build is untouched
```

### 3. Enable GPU per stage

| Stage | How | Notes |
|-------|-----|-------|
| Detection | automatic | `ultralytics.device` (auto = CUDA when available) |
| Stabilization | `--stab-gpu` | works out of the box (default detector is ORB) |
| Georeferencing | `--geo-gpu --geo-detector orb` | ORB is required; RootSIFT is not CUDA-accelerated |

Only ORB is CUDA-accelerated, so georeferencing must be switched to it. Since v1.4.0 `--geo-detector` does this on the command line, with no config copy needed. A copied config is still required if you did **not** add `video` to your OpenCV `BUILD_LIST` (step 1):

```bash
geotrax config copy                       # writes default_copy.yaml
# edit default_copy.yaml:
#   tracker.botsort.gmc_method: none      # ONLY if you did not add 'video' to BUILD_LIST (step 1); disables tracker GMC, verify tracking quality
```

These are the GPU-accelerated analogues of the two reproduce commands in [`data/README.md`](data/README.md). Their outputs are *equivalent, not identical* to the committed `data/results-pixel/` and `data/results-full/`: the CUDA path georeferences with ORB (RootSIFT is not GPU-accelerated), GPU feature extraction is not bit-identical to CPU, and with `gmc_method: none` the tracker runs without motion compensation.

```bash
# Pixel-coordinate results: GPU stabilization, no georeferencing
geotrax batch data/U_video_cut.mp4 --no-geo --show-class-names --show-conf --stab-gpu

# Full pipeline: GPU stabilization + georeferencing (needs the orthophoto/segmentation/master-frame data; see data/README.md)
geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames \
  --show-lanes --plot-segmentations -vm 0 3 --stab-gpu --geo-gpu --geo-detector orb
```

(Append `-c default_copy.yaml` to both if you needed the `gmc_method: none` workaround above.)

### 4. Benchmark: detection-only CUDA vs fully CUDA

To isolate the GPU effect, disable the CPU-bound visualization and plotting (`--no-save --no-show --no-plot-save --no-plot-show`) and wipe a scratch output folder before each run (a clean, non-skipped run that leaves your real results untouched). Do **not** pass `--recompute`: the master→orthophoto homography is a one-time, expensive registration normally cached in the master-frames folder, so leaving it cached keeps the benchmark on the per-video work (detection, stabilization, reference→master registration).

Three configurations are compared:
- **geo-trax default (CPU)**: the shipped defaults with RootSIFT georeferencing; what a CPU-only user actually runs.
- **ORB (CPU)**: the GPU-tuned config on CPU; a controlled, same-config baseline for the GPU run.
- **fully CUDA**: the same ORB config with `--stab-gpu --geo-gpu`.

The default run needs no config file: with no `-c`, geo-trax uses its shipped defaults (RootSIFT georeferencing, CPU for both stabilization and georeferencing). The ORB runs switch the georeferencing detector with `--geo-detector orb`; add `-c default_copy.yaml` to them as well if you needed `tracker.botsort.gmc_method: none`.

With [hyperfine](https://github.com/sharkdp/hyperfine):

```bash
OUT=/tmp/geotrax_bench
COMMON="geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames --no-save --no-show --no-plot-save --no-plot-show -of $OUT"
hyperfine --warmup 1 --runs 5 --prepare "rm -rf $OUT" \
  -n "geo-trax default (CPU, RootSIFT)" "$COMMON" \
  -n "ORB (CPU)"                        "$COMMON --geo-detector orb" \
  -n "fully CUDA (ORB)"                 "$COMMON --geo-detector orb --stab-gpu --geo-gpu"
```

Or without extra tools:

```bash
bench () { local label="$1"; shift; rm -rf /tmp/geotrax_bench
  local t0=$(date +%s.%N)
  geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames \
    --no-save --no-show --no-plot-save --no-plot-show -y -o -of /tmp/geotrax_bench "$@" >/tmp/bench.log 2>&1 \
    && awk -v a="$t0" -v b="$(date +%s.%N)" -v l="$label" 'BEGIN{printf "%-28s %.1f s\n", l, b-a}' \
    || echo "$label FAILED (see /tmp/bench.log)"; }
bench "geo-trax default (CPU)"
bench "ORB (CPU)"        -c default_copy.yaml
bench "fully CUDA (ORB)" -c default_copy.yaml --stab-gpu --geo-gpu
```

Example on an **NVIDIA RTX 4090** (5-second sample clip, 150 frames; hyperfine mean ± σ over 5 runs):

<details>
<summary>Benchmark system</summary>

- **OS**: Ubuntu 24.04.4 LTS (Linux 6.8, x86_64)
- **CPU**: 13th Gen Intel Core i9-13900KF (24 cores / 32 threads)
- **RAM**: 62 GiB
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB, driver 580.159.03)
- **Software**: Python 3.11.5, torch 2.12.1 (CUDA 13.0), OpenCV 4.13.0, stabilo 1.3.0

</details>

| Pipeline | Wall time | vs default |
|----------|-----------|------------|
| CPU stabilization and georeferencing (geo-trax defaults) | 273.7 ± 0.9 s | 1× |
| CPU stabilization and georeferencing (GPU-matched config) | 348.4 ± 0.4 s | 0.8× |
| Fully CUDA (GPU-matched config) | 16.4 ± 2.0 s | **16.7×** |

> Fully CUDA is **16.7× faster than the geo-trax default** and **21.3× faster than the same ORB config on CPU** (row 2). Row 1 is the shipped defaults (RootSIFT georeferencing); rows 2–3 use the GPU-matched config, which switches georeferencing to ORB so the CPU and GPU runs do identical work (only ORB is CUDA-accelerated). Row 2 is slower than row 1 because ORB at a 250k feature ceiling with brute-force matching is costlier on CPU than RootSIFT. Treat these as a **relative** comparison, not absolute throughput: geo-trax's defaults favor maximum accuracy and reliability, with detection at **1920×1920**, stabilization at only a **0.5 downscale** (roughly 2K per frame on this 4K clip) with a high `max_features` ceiling, and RootSIFT georeferencing with a very high `max_features` and conservative MAGSAC++ matcher/projection settings, all against an **8000×8000** orthophoto. Lighter settings would cut absolute times across the board; the point is the CPU→GPU ratio.

</details>

## Usage

The `geotrax` CLI provides one subcommand per stage: `batch` (primary entry point), `extract`, `georeference`, `visualize`, `plot`, `aggregate`, and `config`. Run `geotrax -h` or `geotrax <subcommand> -h` for the full reference (`python -m geotrax` works identically).

```bash
# Recursively process a directory (or a single video) without georeferencing
geotrax batch path/to/videos/ --no-geo

# Run an individual stage on its own
geotrax extract video.mp4                  # detect, track, and stabilize
geotrax visualize video.mp4 --save         # render an annotated video from existing results
geotrax plot video.mp4                     # trajectory and distribution plots
```

> [!TIP]
> See [data/README.md](data/README.md) for sample data and testing examples.

<details>
<summary><b>💡 More Examples & Advanced Usage</b></summary>

```bash
# Use a custom config (bundled preset by name, or a path to your own file)
geotrax batch video.mp4 -c confident
geotrax batch video.mp4 -c path/to/custom_config.yaml

# Fill per-track detection gaps with linear interpolation (adds is_interpolated column to .txt output)
geotrax batch video.mp4 --no-geo --interpolate

# SAHI sliced inference for small objects (requires: pip install 'geo-trax[sahi]')
geotrax batch video.mp4 --no-geo --sahi

# Regenerate visualization without re-running extraction
geotrax batch video.mp4 --viz-only --save

# Show lane IDs, hide the speed overlay (requires georeferencing)
geotrax batch video.mp4 --viz-only --save --show-lanes --hide-speed

# Georeference an already-extracted video against orthophotos
geotrax georeference video.mp4 -orf path/to/orthophotos -mf path/to/master_frames

# Aggregated trajectory plots, excluding buses and trucks
geotrax batch path/to/PROCESSED/ --plot-only --plot-aggregate --plot-class-filter 1 2

# Merge multi-drone results for the same locations into a unified dataset
geotrax aggregate path/to/PROCESSED/

# Rotated box modes (3/4): boxes oriented to vehicle heading, on original (3) or stabilized (4) frame
geotrax visualize video.mp4 --save --viz-mode 3 4
```

<p align="center"><img src="https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization_obb_zoom.jpg" alt="Geo-trax mode 3 rotated bounding boxes, zoomed detail" width="70%"><br><em>Mode 3 rotated bounding boxes, zoomed detail from the same scene as the animation above.</em></p>

<details>
<summary><b>⚠️ Rotated box modes (3 and 4): known limitations</b></summary>

Modes 3 and 4 replace the standard axis-aligned YOLO detections with **rotated bounding boxes**: each box is sized to the vehicle's estimated physical length and width and rotated to align with its travel direction (heading). The heading is derived from the camera-motion-free stabilized trajectory; mode 3 projects the result back onto the original frame, while mode 4 draws directly on the stabilized frame. They are the most informative rendering modes but also the most sensitive to data quality:

- **Size estimation may fail.** Estimates are computed over frames where the vehicle moves in a nearly straight path with sufficient displacement. Short tracks, near-stationary vehicles, or detections close to the frame edges can yield no usable estimate. In those cases the box dimensions fall back to a per-vehicle Q25 aggregate of the raw YOLO bounding box extents (rotated to the heading), which tends to be inflated during turns since axis-aligned detections expand as the vehicle turns. Fallback boxes are rendered with a **dashed outline** so they are easy to identify.
- **Heading may be unreliable.** For very slow or stationary vehicles the motion direction cannot be determined; the box is then aligned with the longer axis of the raw bounding box instead.
- **Back-projection distortion (mode 3 only).** Oriented boxes are computed in stabilized space and projected back onto the original frame via the inverse stabilization homography. Under strong camera motion this can produce visibly skewed boxes.
- **Edge clipping is approximate.** When a vehicle is entering or exiting the frame, the detection only covers its visible part, so the oriented box is clipped to that footprint instead of being drawn at full size (the clip is triggered once the detection reaches within `edge_clip_margin` pixels of the border, since the YOLO box may stop a few pixels short of the true edge). That footprint, however, is only known as an axis-aligned (HBB) detection box, whereas the rendered box is rotated to the heading — so the clip boundary only approximates where the rotated vehicle actually leaves the frame. The clip rectangle is temporally smoothed (`edge_clip_smoothing`) so the box shrinks steadily as the vehicle exits rather than jumping with per-frame detection noise.
- Both modes require that the extraction stage was run with stabilization enabled (`stabilize: true` in the config).

</details>

> [!NOTE]
> **Why use master frames?** When georeferencing, geo-trax can route each video's homography through a shared *master frame* per location ID. A master frame is a high-quality, near-nadir BEV frame chosen once per location (see [`tools/find_master_frames.py`](tools/find_master_frames.py)), used instead of registering every video's reference frame directly to the orthophoto. The mapping is split into two homographies: `reference → master` (recomputed per video) and `master → orthophoto` (computed **once per location ID and cached**, validated by a hash of the master image). This gives two benefits:
> - **Speed**: the expensive cross-domain `master → orthophoto` registration runs once and is reused across every drone and flight at that location, instead of once per video.
> - **Consistency & robustness**: every video is matched against the *same* master frame. This same-modality BEV-to-BEV registration is far more reliable than a direct BEV-to-orthophoto match, so trajectories from different drones, altitudes, and viewpoints resolve into one coherent coordinate system.
>
> Master frames are enabled by default. Disable them with `--no-master`, or force re-computation of the cached `master → orthophoto` homography with `--recompute`.

</details>

<details>
<summary><b>📁 Output file formats</b></summary>

Suppose the input video is `video_file.mp4`. By default, outputs are written to a `results/` sub-folder next to the input; the folder and all filename postfixes are configurable via the `output:` section of the pipeline config (or `--output-folder` / `-of` for the folder).

- **video_file.txt** (`<stem><tracks_postfix>.txt`): Contains the extracted vehicle trajectories in the following format:

  ```text
  frame_id, vehicle_id, x_c(unstab), y_c(unstab), w(unstab), h(unstab), x_c(stab), y_c(stab), w(stab), h(stab), class_id, confidence, vehicle_length, vehicle_width
  ```

    where:
  - `frame_id`: Frame number (0, 1, ...).
  - `vehicle_id`: Unique vehicle identifier (1, 2, ...).
  - `x_c(unstab)`, `y_c(unstab)`: Unstabilized vehicle centroid coordinates.
  - `w(unstab)`, `h(unstab)`: Unstabilized vehicle bounding box width and height.
  - `x_c(stab)`, `y_c(stab)`: Stabilized vehicle centroid coordinates.
  - `w(stab)`, `h(stab)`: Stabilized vehicle bounding box width and height.
  - `class_id`: Vehicle class identifier (0: car (incl. vans), 1: bus, 2: truck, 3: motorcycle)
  - `confidence`: Detection confidence score (0-1).
  - `vehicle_length`, `vehicle_width`: Estimated vehicle dimensions in pixels.
  - `is_interpolated` *(optional, 15th column)*: Present only when `extraction.interpolate: true` (CLI: `--interpolate`). `0` = real detection, `1` = linearly interpolated to fill a frame gap. Gaps larger than the active tracker's `track_buffer` are left unfilled (the tracker would not persist a lost track's ID across a longer occlusion).

- **video_file_vid_transf.txt** (`<stem><stab_transform_postfix>.txt`): Contains the transformation matrix for each frame in the format:

  ```text
  frame_id, h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

    where:
  - `frame_id`: Frame number of the stabilized frame (starts from `cut_frame_left + 1` since the reference frame itself has no transform).
  - `hij`: Elements of the 3x3 homography matrix that maps each frame (`frame_id`) to the reference frame.

- **video_file.yaml**: Video metadata and the configuration actually used to process `video_file.mp4` — the config file's values with every CLI flag and `--set` override already applied, so the file is a faithful record of the run rather than of the config it started from.

- **video_file_mode_X.mp4** (`<stem><visualization_postfix>_mode_<X>.mp4`): Annotated video in five rendering modes (X = 0 / 1 / 2 / 3 / 4):
  - **Mode 0**: overlaid on the original (unstabilized) video
  - **Mode 1**: overlaid on the stabilized video
  - **Mode 2**: plotted on the static reference frame
  - **Mode 3**: rotated bounding boxes on the original video, where each box is sized to the vehicle's estimated physical dimensions and rotated to its per-frame heading (derived from the camera-motion-free stabilized trajectory and projected back onto the original frame). Requires stabilization to have been run.
  - **Mode 4**: the same rotated bounding boxes as Mode 3, but drawn directly on the stabilized video (no back-projection). Requires stabilization to have been run.

  Each version can display vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement. If an input `video_file.csv` file is available in the same directory as the input video, i.e., the converted flight logs, vehicle speed and lane information can also be displayed.

- **video_file.csv** (`<stem><georeferenced_postfix>.csv`): Contains the georeferenced vehicle trajectories in a tabular format. This file includes both geographic and local coordinates, estimated real-world dimensions, kinematic data, road section, and lane information. The columns are:

  ```text
  Vehicle_ID, [Timestamp,] Frame_Number, Ortho_X, Ortho_Y, Local_X, Local_Y, Latitude, Longitude, Vehicle_Length, Vehicle_Width, Vehicle_Class, Vehicle_Speed, Vehicle_Acceleration, Road_Section, Lane_Number, Visibility[, Is_Interpolated]
  ```

    where:
  - `Vehicle_ID`: Unique vehicle identifier.
  - `Timestamp`: Timestamp of the frame (YYYY-MM-DD HH:MM:SS.ms). Present only when a flight-log CSV with timestamps is available alongside the video.
  - `Frame_Number`: Video frame index corresponding to this detection.
  - `Ortho_X`, `Ortho_Y`: X and Y coordinates of the vehicle centroid in the orthophoto's pixel coordinate system.
  - `Local_X`, `Local_Y`: X and Y coordinates of the vehicle centroid in a local projected coordinate system (e.g., EPSG:5186 for KGD2002 / Central Belt 2010 used in the Songdo experiment).
  - `Latitude`, `Longitude`: Geographic coordinates of the vehicle centroid (WGS84).
  - `Vehicle_Length`, `Vehicle_Width`: Estimated vehicle dimensions in meters.
  - `Vehicle_Class`: Vehicle class identifier (0: car (incl. vans), 1: bus, 2: truck, 3: motorcycle).
  - `Vehicle_Speed`: Estimated vehicle speed in km/h.
  - `Vehicle_Acceleration`: Estimated vehicle acceleration in m/s$^2$.
  - `Road_Section`: Identifier for the road segment the vehicle is on.
  - `Lane_Number`: Identifier for the lane the vehicle is in.
  - `Visibility`: Boolean indicating if the vehicle's bounding box is fully visible within the frame.
  - `Is_Interpolated` *(optional)*: Present only when extraction was run with `--interpolate` (`extraction.interpolate: true`). `0` = real detection, `1` = row synthesized by linear interpolation at the extraction stage to fill a frame gap; propagated from the `.txt` tracks file.

- **video_file_geo_transf.txt** (`<stem><geo_transform_postfix>.txt`): Contains the 3x3 georeferencing transformation matrix (homography) that maps points from the video's reference frame to the orthomap. The format is a comma-separated list of the 9 matrix elements:

  ```text
  h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

**Note:** *All output files are saved in the configured output folder (default: `results/` sub-folder next to the input video); nothing is written next to the input video, so the source dataset can stay read-only. Trajectory and distribution plots are always written to a `plots/` sub-folder inside the output folder.*

</details>

## Real-World Deployment: The Songdo Experiment

Geo-trax was validated in a large-scale urban traffic monitoring campaign in Songdo, South Korea, where it processed footage from a fleet of 10 drones to produce the [**Songdo Traffic**](https://doi.org/10.5281/zenodo.13828383) dataset. The detection model was trained on the companion [**Songdo Vision v1**](https://doi.org/10.5281/zenodo.13828408) dataset (a later [v2](https://doi.org/10.5281/zenodo.21728430) annotation-quality release exists on Zenodo; the released model has not been retrained on it). The dataset is described in the [publication](#citation).

| Songdo campaign | |
|---|---|
| 📍 Location | Songdo International Business District, South Korea |
| 📅 Duration | 4 days (October 4 to 7, 2022) |
| 🚁 Fleet | 10 drones (DJI Mavic 3), 140 to 150 m altitude, 4K at 29.97 fps |
| 🔭 Coverage | 20 busy intersections |
| 🚗 Result | ~700,000 georeferenced vehicle trajectories |

🎥 *Demo of Geo-trax applied to the Songdo experiment:* [https://youtu.be/gOGivL9FFLk](https://youtu.be/gOGivL9FFLk)

The blocks below document the project layout and data-wrangling workflow used in that campaign; they double as the recommended setup for your own multi-drone projects.

<details>
<summary><b>📂 Recommended project folder structure</b></summary>

The layout below mirrors the Songdo experiment and matches the pipeline's auto-detection defaults, letting `geotrax batch` run with no path flags. Two conventions do the heavy lifting:

- **A `PROCESSED/` folder anchors auto-detection.** When georeferencing or plotting needs orthophotos, master frames, or segmentations and no explicit path is given, Geo-trax walks *up* from the video until it finds `PROCESSED`, then looks for a sibling `ORTHOPHOTOS/` folder.
- **A location ID ties each video to its assets.** The location ID is the leading letters in the clip filename (`A1.mp4` → `A`), so `A1.mp4` automatically resolves to `ORTHOPHOTOS/A.png`, `ORTHOPHOTOS/master_frames/A.png`, and `ORTHOPHOTOS/segmentations/A.csv`.

### Directory tree

```text
<project>/                                 # project root (name arbitrary)
├── RAW/                                   # untouched drone footage + flight logs (never modified)
│   └── 2022-10-07/D1/PM1/                 # arbitrary nesting, e.g. date / drone / session
│       ├── DJI_0001.MP4  DJI_0001.SRT
│       └── DJI_0002.MP4  DJI_0002.SRT     # drone splits a recording into segments (file-size limit)
├── PROCESSED/                             # pipeline input (auto-detect anchor)
│   └── 2022-10-07/D1/PM1/
│       ├── 0_merged.mp4  0_merged.srt     # merged flight video + log (temporary, deletable)
│       ├── 0_merged.txt                   # cut list: start/end frames, one cut per line (temporary)
│       ├── A1.mp4  A1.csv                 # cut clip + flight log; 'A' = location ID, '1' = sequence
│       ├── A2.mp4  A2.csv                 # next clip at the same location
│       └── results/                       # pipeline outputs, written next to each clip
│           ├── A1.yaml                    # run metadata: the configuration actually used
│           ├── A1.txt                     # pixel-coordinate tracks
│           ├── A1_vid_transf.txt          # stabilization homographies
│           ├── A1_geo_transf.txt          # georeferencing homography
│           ├── A1.csv                     # georeferenced trajectories + kinematics
│           ├── A1_mode_0.mp4              # video with overlaid boxes & trajectories (modes 0/1/2/3/4)
│           └── plots/                     # various trajectory & distribution plots
├── ORTHOPHOTOS/                           # auto-detected sibling of PROCESSED / DATASET
│   ├── A.png                              # orthophoto cut-out, per location
│   ├── A.txt  (or A.tif)                  # georeferencing parameters (or a georeferenced GeoTIFF)
│   ├── ortho_parameters.txt               # (alternative) shared params + per-location A_center.txt
│   ├── master_frames/                     # optional; consistent reference frame per location
│   │   ├── A.png                          #   reference frame image
│   │   └── A.txt                          #   cached master->ortho homography
│   └── segmentations/                     # optional; per-location lane/road geometry
│       ├── A.csv                          #   lane & road-section polygons
│       └── A.png                          #   overlay image (used for plotting only)
└── DATASET/                               # `geotrax aggregate` output (sibling of PROCESSED)
    └── 2022-10-07_A/                      # one intersection-day
        ├── 2022-10-07_A_AM1.csv           # one CSV per flight session (AM1-AM5, PM1-PM5),
        └── 2022-10-07_A_PM1.csv           #   trajectories merged across drones for that session
```

`RAW/` is kept immutable; everything downstream lives under `PROCESSED/`. The `master_frames/` and `segmentations/` sub-folders are optional; provide them only when you need cross-flight georeferencing consistency or lane-level analysis. `DATASET/` is created by `geotrax aggregate` and is also a valid auto-detection anchor for `ORTHOPHOTOS/`.

</details>

<details>
<summary><b>🏷️ Clip naming conventions</b></summary>

Only the **leading location letters** of a clip filename are required by the code (parsed by `determine_location_id`). The contextual metadata (date, drone, session) normally lives in the **folder path**, so each clip can be named compactly as location ID + sequence number:

```text
2022-10-07/D10/PM5/U1.mp4
│          │   │   └── clip: location ID 'U' + sequence number '1'
│          │   └────── flight session: AM1-AM5 (morning) / PM1-PM5 (afternoon)
│          └────────── drone ID (D1, D2, ...)
└───────────────────── capture date (ISO 8601, YYYY-MM-DD)
```

These compact names are assigned automatically by the cutting step, not typed by hand: given a location map (a JSON file pairing each label with its `[lat, lon]` center), [`tools/cut_merged_videos_and_logs.py`](tools/cut_merged_videos_and_logs.py) labels every clip with the location nearest to its GPS centroid and appends a per-location sequence number (`U1`, `U2`, ...).

Because only the leading letters matter, the same context can instead be packed into a single self-contained filename when clips are detached from this tree. This is how the sample videos published on Zenodo are named, e.g. `U_D10_2022-10-07_PM5_60s.mp4` (location `U`, drone `D10`, date `2022-10-07`, session `PM5`). Here the per-location sequence number is replaced by a time marker showing where the clip falls within the session: `60s` denotes the first 60 seconds of that session at the location. Either way, the code still extracts location `U`.

| Clip filename | Location ID | Resolves to |
|---|---|---|
| `U1.mp4` | `U` | `ORTHOPHOTOS/U.png`, `master_frames/U.png`, `segmentations/U.csv` |
| `U2.mp4` | `U` | `ORTHOPHOTOS/U.png`, … |
| `U_D10_2022-10-07_PM5_60s.mp4` | `U` | `ORTHOPHOTOS/U.png`, … |

`geotrax aggregate` groups results by location (and date/session), merging clips from different drones that cover the same place into a unified dataset.

</details>

<details>
<summary><b>🛠️ From raw footage to trajectories</b></summary>

The `tools/` directory provides the wrangling scripts that take you from raw footage to pipeline-ready clips (see [`tools/README.md`](tools/README.md) for the full index):

1. **Merge** the recorded video segments and their logs into one video + log per flight session → [`tools/merge_videos_and_logs.py`](tools/merge_videos_and_logs.py)
2. **Cut** each merged flight into per-location clips: list the start/end frames of each stable hover in `0_merged.txt`, then split (converting the DJI SRT log to a per-clip CSV) → [`tools/cut_merged_videos_and_logs.py`](tools/cut_merged_videos_and_logs.py)
3. **QA / repair** the cut logs → [`tools/find_cut_video_issues.py`](tools/find_cut_video_issues.py), [`tools/fix_timestamp_anomalies.py`](tools/fix_timestamp_anomalies.py), [`tools/interpolate_missing_timestamps.py`](tools/interpolate_missing_timestamps.py)
4. **Build the georeferencing assets**: orthophoto cut-outs per location → [`tools/subset_orthophoto.py`](tools/subset_orthophoto.py); master frames → [`tools/find_master_frames.py`](tools/find_master_frames.py); lane segmentations are drawn manually, with overlays rendered via [`tools/viz_segmentations.py`](tools/viz_segmentations.py)
5. **Run the pipeline**: `geotrax batch PROCESSED/ ...`; orthophotos, master frames, and segmentations are auto-detected from the sibling `ORTHOPHOTOS/` folder.
6. **(Optional) Aggregate** results across drones and flights for the same location → `geotrax aggregate PROCESSED/`, which writes a unified dataset to a sibling `DATASET/` folder.

### Lessons from the Songdo experiment

- Treat `RAW/` as read-only archival storage and derive everything under `PROCESSED/`; the wrangling steps are reproducible from the raw footage.
- The **master frame** is an intermediary coordinate system per location: aligning every flight to one shared reference frame keeps trajectories from different drones, altitudes, and viewpoints in a single consistent coordinate system.
- Coordinates were projected to a local CRS (EPSG:5186, KGD2002 / Central Belt 2010) alongside WGS84 lat/lon; set your own CRS in the `georef:` config section.
- Imagery was captured at ~140–150 m altitude in 4K, giving a ground sampling distance of ≈ 0.027 m/px (the default `extraction.gsd`). Re-tune the GSD for different altitudes or cameras.

</details>

## Citation

If you use **Geo-trax** in your research or software, please cite:

1. **Journal article** (preferred for any use of the framework):

    ```bibtex
    @article{fonod2025advanced,
      title = {Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery},
      author = {Fonod, Robert and Cho, Haechan and Yeo, Hwasoo and Geroliminis, Nikolas},
      journal = {Transportation Research Part C: Emerging Technologies},
      volume = {178},
      pages = {105205},
      year = {2025},
      publisher = {Elsevier},
      doi = {10.1016/j.trc.2025.105205},
      url = {https://doi.org/10.1016/j.trc.2025.105205}
    }
    ```

2. **Software archive** (when referencing or building on the code itself):

    ```bibtex
    @software{fonod2026geo-trax,
      author = {Fonod, Robert},
      title = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
      year = {2026},
      month = aug,
      version = {1.4.1},
      doi = {10.5281/zenodo.12119542},
      url = {https://github.com/rfonod/geo-trax},
      license = {MIT}
    }
    ```

## Contributions

Early code received key contributions from [Haechan Cho](https://github.com/cho-96) (georeferencing) and [Sohyeong Kim](https://github.com/shgold) (video/flight-log merging). Community contributions are welcome: open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) for more details.
