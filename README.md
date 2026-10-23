# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![PyPI - Version](https://img.shields.io/pypi/v/geo-trax)](https://pypi.org/project/geo-trax/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/geo-trax?label=total%20downloads)](https://pepy.tech/project/geo-trax) [![CI](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![Hugging Face](https://img.shields.io/badge/🤗%20Model-rfonod%2Fgeo--trax-yellow)](https://huggingface.co/rfonod/geo-trax) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/rfonod/geo-trax) [![Project Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax) [![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk) [![Geo Export](https://img.shields.io/badge/Geo%20Export-GeoJSON%20%7C%20GeoPackage-4c8)](https://rfonod.github.io/geo-trax/guides/export/) [![Docs](https://img.shields.io/badge/docs-rfonod.github.io%2Fgeo--trax-blue)](https://rfonod.github.io/geo-trax/)

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

**Optional dependency groups** (development/testing tools, ONNX export, SAHI sliced inference, documentation site):

```bash
python -m pip install -e '.[dev]'      # development + test tooling
python -m pip install -e '.[export]'   # ONNX export dependencies
python -m pip install -e '.[sahi]'     # SAHI sliced inference for small-object detection (--sahi)
python -m pip install -e '.[docs]'     # documentation site: mkdocs serve / mkdocs build --strict
# uv pip install -e '.[dev]'           # uv equivalents
# poetry install --extras dev          # Poetry equivalents
# poetry install --extras export
# poetry install --extras sahi
# poetry install --extras docs
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
geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames --show-lanes

# Scale up: process a whole project tree, then merge multi-drone results into one dataset
geotrax batch path/to/PROCESSED/
geotrax aggregate path/to/PROCESSED/
```

Run `geotrax -h` or `geotrax batch -h` for all options. The scale-up commands above run flag-free with the [recommended project structure](#real-world-deployment-the-songdo-experiment); any other layout works with explicit path flags. To georeference your own footage, [Your own data](https://rfonod.github.io/geo-trax/guides/own-data/) lists the files each location needs and the defaults to change outside South Korea.

<details>
<summary><b>📋 Full Feature Overview</b></summary>

- **Detection**: YOLOv8s on aerial BEV imagery; detects car (incl. vans), bus, truck, and motorcycle; optional per-class confidence thresholds (`extraction.class_conf`) and [SAHI](https://github.com/obss/sahi) sliced inference for improved small-object recall (`--sahi`).
- **Tracking**: six multi-object trackers (BoT-SORT default; see [Tracking](#tracking)); optional per-track frame-gap interpolation.
- **Stabilization**: homography-based trajectory correction via [Stabilo](https://github.com/rfonod/stabilo) ⚖️, tuned with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯; optional CUDA acceleration (`--stab-gpu`).
- **Georeferencing**: frame-to-orthophoto registration; outputs lat/lon, local CRS, speed, acceleration, and lane assignment per vehicle; optional CUDA acceleration (`--geo-gpu`).
- **Visualization**: track overlays on original, stabilized, or static-reference video, in five rendering modes (incl. oriented bounding boxes).
- **Analysis**: trajectory maps, kinematic distributions, and class/dimension charts, per-video or aggregated across drones and sessions; GIS export (GeoPackage/GeoJSON) of trajectories for QGIS, ArcGIS, or kepler.gl (`geotrax export`).
- **Scaling & tooling**: batch-processes directory trees and aggregates multi-drone data; includes standalone utilities for end-to-end data preparation, training, evaluation, and validation.

</details>

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- Modularized, OOP-based pipeline with custom reference frame support and georeferencing leveraging Stabilo's image-matching backend.
- Simulation and scenario export: `geotrax export --format sumo` (and/or OpenDRIVE, CommonRoad) to turn an extracted dataset into a drop-in scenario for traffic simulators and digital twins, reusing the lane and road-section geometry already carried by the orthophoto segmentations.
- Live/online mode: ingest RTMP-style drone feeds for real-time georeferenced output, for live monitoring and incident detection rather than offline batches.
- Batch inference and multi-thread processing, plus distributed fan-out (Ray, Dask, or Slurm) so `geotrax batch` scales across a cluster for multi-drone, multi-intersection campaigns.
- Real-world map visualization (e.g., MovingPandas, contextily) and an interactive web viewer with map-based playback and filtering by class, lane, speed, and time, so datasets can be explored without writing plotting code. (`geotrax export` already produces GeoPackage/GeoJSON layers that open directly in QGIS or kepler.gl.)

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
geotrax batch video.mp4 -c confident                     # a bundled preset, by name
geotrax config copy                                      # editable copies: <name>_copy.yaml
geotrax batch video.mp4 -c default_copy.yaml             # your edited copy
geotrax batch video.mp4 --set conf=0.35 --set iou=0.6    # one-off overrides of any key
```

The [Configuration guide](https://rfonod.github.io/geo-trax/guides/configuration/) covers inspecting and copying configs, the `--set` rules and their interplay with CLI flags, the run metadata, logs, and the update check; the [config reference](https://rfonod.github.io/geo-trax/reference/config/) documents every key.

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

To use a different model, point `--model` (CLI) or `extraction.model` (config) to a local `.pt` path or `hf://<org>/<repo>/<file>.pt`; any [Ultralytics](https://github.com/ultralytics/ultralytics)-compatible model works. An `hf://` reference follows the repo's current `main`; to pin an exact version, add a revision (commit SHA, tag or branch) after the repo name: `hf://<org>/<repo>@<revision>/<file>.pt`. Class filtering and renaming are covered in the [Detection and tracking guide](https://rfonod.github.io/geo-trax/guides/detection-tracking/).

### Per-Class Confidence Thresholds

A single `ultralytics.conf` threshold rarely suits every class: rare or harder classes (buses, trucks) often deserve a lower bar than abundant cars. `extraction.class_conf` sets a threshold per class ID; classes it does not list keep the global `conf`:

```bash
geotrax extract video.mp4 --set 'class_conf={1: 0.15, 2: 0.4}'   # or set extraction.class_conf in a copied config
```

Each detection is held to the threshold of its own class *before* it reaches the tracker, so rejected boxes never start or extend a track; with `class_conf: null` (the default) extraction is unchanged. The [Detection and tracking guide](https://rfonod.github.io/geo-trax/guides/detection-tracking/#confidence-thresholds) explains how the thresholds interact with the per-track class vote and with `ultralytics.max_det`.

### Small-Object Detection with SAHI

For footage where vehicles are near the detection floor (higher altitudes, sub-4K sensors), the extraction stage can optionally run [SAHI](https://github.com/obss/sahi) sliced inference: each frame is split into overlapping slices, the detector runs on every slice (plus one full-frame pass), and the results are merged before tracking. This substantially improves tiny-object recall, at a cost that grows with the number of slices. On 4K footage the default 1920 x 1080 slices with 20% overlap form a 3 x 3 grid, because two overlapping slices cover only 3456 of the 3840 px; that is nine passes plus the full-frame pass, roughly 10x the per-frame detection cost. Slightly larger slices give a 2 x 2 grid at about half that cost; see the [SAHI guide](https://rfonod.github.io/geo-trax/guides/sahi/#slice-grid).

```bash
geotrax extract video.mp4 --sahi     # also available on 'geotrax batch'
geotrax extract video.mp4 --sahi --set sahi.slice_width=2160 --set sahi.slice_height=1215   # 2 x 2 grid on 4K
```

SAHI is an optional extra (`python -m pip install 'geo-trax[sahi]'`); without it, the run stops at once with the install command for your setup. It supports YOLO models only and cannot be combined with the `tracktrack` tracker or with ReID model `auto`. The [SAHI guide](https://rfonod.github.io/geo-trax/guides/sahi/) covers the slicing and merge settings and which detection settings still apply.

### Custom Model Training

Training and export scripts for custom YOLO detectors live in `train/`, with a SLURM wrapper for HPC clusters. See [train/README.md](train/README.md).

## Tracking

Six multi-object trackers from [Ultralytics](https://github.com/ultralytics/ultralytics) are available: **BoT-SORT** (`botsort`, the default), **ByteTrack** (`bytetrack`), **OC-SORT** (`ocsort`), **Deep OC-SORT** (`deepocsort`), **FastTracker** (`fasttrack`) and **TrackTrack** (`tracktrack`). Selection is config-driven, with no code changes:

```bash
geotrax extract video.mp4 --set tracker.active=bytetrack
```

The [Detection and tracking guide](https://rfonod.github.io/geo-trax/guides/detection-tracking/#trackers) compares them (accuracy, speed, appearance ReID, in-tracker camera-motion compensation) and covers tuning their parameters. For a head-to-head comparison on your own data, see [`tools/compare_tracking.py`](tools/compare_tracking.py).

## Stabilization

Quasi-stationary drone footage is never perfectly static: wind, GPS drift, and gimbal corrections shift the frame throughout a flight. Geo-trax compensates by registering every frame back to a fixed reference frame with a homography, via [Stabilo](https://github.com/rfonod/stabilo) ⚖️. Tracks are written in both raw and stabilized image coordinates, and the per-frame 3 × 3 transform is saved alongside them.

Registration quality depends on the feature detector. The default is `orb`, tuned for high-altitude BEV footage with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯 and used for every published result. Five other classical OpenCV detectors and five learning-based kornia models are available through `--stab-detector`; the [Feature detectors guide](https://rfonod.github.io/geo-trax/guides/feature-detectors/) compares them and explains the rotation and memory caveats of the learned ones before you switch.

Masking, RANSAC, and downsampling parameters live under `cfg -> stabilo`. To skip stabilization entirely, set `extraction.stabilize: false`.

## Georeferencing

The georeferencing stage turns stabilized pixel trajectories into real-world ones by registering the video's reference frame against a north-up **orthophoto** of the same location. Output is a per-vehicle CSV with WGS84 coordinates, a local projected CRS, speed, acceleration, and lane / road-section assignment.

This reuses the image-matching machinery from stabilization, but defaults to **`rsift`** (RootSIFT) instead of `orb`. Registering a drone frame against an orthophoto is a cross-domain match, often across years, seasons, and lighting conditions, and is considerably harder than matching consecutive video frames:

```bash
geotrax georeference video.mp4 -orf data/orthophotos --geo-detector sift
```

Orthophotos are required; lane/road-section segmentations and per-location master frames are optional and improve lane assignment and robustness (see the [georeferencing guide](https://rfonod.github.io/geo-trax/guides/georeference/) and [data/README.md](data/README.md) for how to obtain them). Matching parameters live under `cfg -> georef -> matching`, kinematic filtering under `cfg -> georef -> filtering`.

> ⚠️ Outside South Korea, set `georef.transformation.target_crs` to a projected CRS for your area (e.g. its UTM zone). The shipped EPSG:5186 is the Songdo one, and speeds, accelerations and dimensions are measured in it. [Your own data](https://rfonod.github.io/geo-trax/guides/own-data/) describes this and the file formats for a new location.

> The [learning-based detectors](https://rfonod.github.io/geo-trax/guides/feature-detectors/#learning-based-detectors) are a poor fit for this stage in particular: the orthophoto is north-up while the drone frame can be at any heading, and every learned model except `keynet` is upright-only.

## GPU acceleration

Object **detection** already runs on CUDA automatically whenever a compatible GPU and PyTorch build are present (via the `ultralytics.device` config key, auto by default). The **stabilization** (`--stab-gpu`) and **georeferencing** (`--geo-gpu`) image-matching steps can *optionally* be CUDA-accelerated too, through [Stabilo](https://github.com/rfonod/stabilo). This needs a CUDA-enabled OpenCV build and is Linux/Windows only. Stabilo accelerates the **ORB** detector only, so `--geo-gpu` additionally requires `--geo-detector orb`; there is no CPU fallback, so requesting GPU without a working CUDA device raises an error.

> The learning-based detectors use a separate torch device (`--stab-device` / `--geo-device`) instead; see [Feature detectors](https://rfonod.github.io/geo-trax/guides/feature-detectors/#acceleration). The two cannot be combined.

The full setup (building OpenCV with CUDA, installing geo-trax without overwriting that build, and enabling the GPU per stage) is in the [GPU acceleration guide](https://rfonod.github.io/geo-trax/guides/gpu/), together with an RTX 4090 benchmark (16.7× faster than the CPU defaults on the sample clip).

## Usage

The `geotrax` CLI provides one subcommand per stage: `batch` (primary entry point), `extract`, `georeference`, `visualize`, `plot`, `aggregate`, `export`, and `config`. Run `geotrax <subcommand> -h` for its options (`python -m geotrax` works identically).

<a id="geo-export"></a>
The [documentation site](https://rfonod.github.io/geo-trax/) has a guide per stage, plus guides to [configuration](https://rfonod.github.io/geo-trax/guides/configuration/) and to preparing [your own data](https://rfonod.github.io/geo-trax/guides/own-data/). The stage guides have more examples, the [output file formats](https://rfonod.github.io/geo-trax/guides/extract/#outputs), the [rotated-box visualization modes](https://rfonod.github.io/geo-trax/guides/visualize/#rotated-boxes-modes-3-and-4), [master frames](https://rfonod.github.io/geo-trax/guides/georeference/#master-frames), and [how to view the GIS exports](https://rfonod.github.io/geo-trax/guides/export/#viewing-exported-files) in the browser, QGIS, or Python. The site's [CLI reference](https://rfonod.github.io/geo-trax/reference/cli/) and [config reference](https://rfonod.github.io/geo-trax/reference/config/) are generated from the code and the shipped config. See [data/README.md](data/README.md) for sample data, including a [sample GIS export](data/README.md#gis-export) of the demo clip.

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

`geotrax aggregate` groups results by location (and date/session), merging clips from different drones that cover the same place into a unified dataset. It reads the grouping from the `<date>/D<n>/<session>/<output folder>/` layout, so drone folders must be named `D` plus a number (results elsewhere are skipped with a warning) and the output folder must be a relative name, not an absolute path. Rows whose timestamp is undefined (frames missing from the flight log) are dropped with a warning; the rest of the clip is kept.

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
  month = oct,
  version = {1.5.2},
  license = {MIT},
  doi = {10.5281/zenodo.12119542},
  url = {https://github.com/rfonod/geo-trax}
}
```

## Contributions

Early code received key contributions from [Haechan Cho](https://github.com/cho-96) (georeferencing) and [Sohyeong Kim](https://github.com/shgold) (video/flight-log merging). Community contributions are welcome: open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) for more details.
