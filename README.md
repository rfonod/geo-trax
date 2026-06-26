# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![PyPI - Version](https://img.shields.io/pypi/v/geo-trax)](https://pypi.org/project/geo-trax/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/geo-trax?label=total%20downloads)](https://pepy.tech/project/geo-trax) [![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/geo-trax?color=%234c1)](https://pypi.org/project/geo-trax/) [![CI](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.9--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/issues) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![Hugging Face](https://img.shields.io/badge/🤗%20Model-rfonod%2Fgeo--trax-yellow)](https://huggingface.co/rfonod/geo-trax) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/rfonod/geo-trax) [![Project Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax) [![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk)

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is a comprehensive pipeline that extracts high-accuracy, georeferenced vehicle trajectories from high-altitude drone imagery. Built for quasi-stationary aerial monitoring of urban traffic, it turns raw bird's-eye view (BEV) drone footage into precise, real-world vehicle trajectories. The framework combines YOLO detection, multi-object tracking, and video stabilization with a robust orthophoto-based georeferencing stage, producing GNSS-tagged, lane-resolved trajectories that are spatially and temporally consistent and ready for large-scale traffic analysis and simulation. It is optimized for urban intersections and arterial corridors, where high-fidelity, vehicle-level insights drive intelligent transportation systems and digital twin applications.

![Geo-trax Output Visualization](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization.webp)

🎬 An accelerated preview of Geo-trax's capabilities. Watch the full ~4 min 4K demo on [YouTube](https://youtu.be/gOGivL9FFLk).

> [!TIP]
> **Just want to see it work?** Try the [interactive demo on 🤗 Hugging Face Spaces](https://huggingface.co/spaces/rfonod/geo-trax) — run the vehicle detector on your own aerial image or short clip right in the browser, no install required.

### Why Geo-trax

- 🛰️ **Real-world output**: georeferenced, lane-resolved trajectories (WGS84 + local CRS) with per-vehicle speed, acceleration, and estimated dimensions, straight from raw BEV drone video.
- 🎯 **Accurate detection**: [YOLOv8s vehicle detector](#detection-model) reaching **0.951 mAP@50**, trained on more than 19,000 annotated aerial images.
- 🚗 **Flexible tracking**: four vehicle classes and [six selectable multi-object trackers](#tracking) (BoT-SORT, ByteTrack, OC-SORT, and more).
- 🌀 **Drone-motion robust**: homography-based stabilization ([Stabilo](https://github.com/rfonod/stabilo)) plus orthophoto image registration for consistent, cross-flight coordinates.
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

Python 3.9 to 3.13. Also works with [uv](https://docs.astral.sh/uv/) (`uv pip install geo-trax`) and [conda](https://www.anaconda.com/docs/getting-started/miniconda/install). For development:

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

**Optional dependency groups** (development/testing tools, ONNX export):

```bash
python -m pip install -e '.[dev]'      # development + test tooling
python -m pip install -e '.[export]'   # ONNX export dependencies
# uv pip install -e '.[dev]'           # uv equivalents
# poetry install --extras dev          # Poetry equivalents
# poetry install --extras export
```

</details>

## Quick Start

`data/U_video_cut.mp4` is a 5-second sample clip included for immediate testing. See [data/README.md](data/README.md) for matching orthophotos.

```bash
# Pixel-coordinate trajectories (no orthophoto required)
geotrax batch data/U_video_cut.mp4 --no-geo --save

# Full pipeline: detection, tracking, stabilization, georeferencing, and visualization
geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -mf data/master_frames --save --show-lanes

# Scale up: process a whole project tree, then merge multi-drone results into one dataset
geotrax batch path/to/PROCESSED/
geotrax aggregate path/to/PROCESSED/
```

Run `geotrax -h` or `geotrax batch -h` for all options. The scale-up commands above run flag-free with the [recommended project structure](#real-world-deployment-the-songdo-experiment); any other layout works with explicit path flags.

<details>
<summary><b>📋 Full Feature Overview</b></summary>

- **Detection**: YOLOv8s on aerial BEV imagery; detects car (incl. vans), bus, truck, and motorcycle.
- **Tracking**: six multi-object trackers (BoT-SORT default); see [Tracking](#tracking) for a comparison.
- **Stabilization**: homography-based trajectory correction via [Stabilo](https://github.com/rfonod/stabilo) 🌀, tuned with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯.
- **Georeferencing**: frame-to-orthophoto registration; outputs lat/lon, local CRS, speed, acceleration, and lane assignment per vehicle.
- **Visualization**: track overlays on original, stabilized, or static-reference video, in three rendering modes.
- **Analysis**: trajectory maps, kinematic distributions, and class/dimension charts, per-video or aggregated across drones and sessions.
- **Scaling & tooling**: batch-processes directory trees and aggregates multi-drone data; includes standalone utilities for end-to-end data preparation, training, evaluation, and validation.

</details>

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- Comprehensive documentation in a dedicated `docs/` folder. A [`tools/README.md`](tools/README.md) index already covers the auxiliary scripts.
- Modularized, OOP-based pipeline with custom reference frame support and georeferencing leveraging Stabilo's image-matching backend.
- Per-class confidence thresholds and oriented bounding box visualization (using azimuth and dimension estimates).
- Trajectory interpolation and SAHI-based small-object detection.
- Batch inference, GPU-accelerated image registration, and multi-thread processing.
- Real-world map visualization (e.g., MovingPandas, contextily) and interactive web app.

</details>

<details>
<summary><b>🔗 Related Projects</b></summary>

Geo-trax integrates with and complements several specialized tools:

- **[Stabilo](https://github.com/rfonod/stabilo) 🌀**: Python library for video and trajectory stabilization using robust homography transformations. Supports various feature detectors, RANSAC algorithms, and user-defined masks. Used as Geo-trax's core stabilization engine.

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

## Detection Model

The default detector is **YOLOv8s** (HBB, 1920 × 1920 px, ~11 M parameters), trained on more than 19,000 annotated aerial images (~679k labeled vehicle instances) and fine-tuned on a curated, high-quality subset. It is hosted on [🤗 Hugging Face](https://huggingface.co/rfonod/geo-trax) and **downloads automatically on first use**. Results on the Songdo Vision test split (1,084 images; full results in [Table 3](https://doi.org/10.1016/j.trc.2025.105205)):

| ID | Label | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| 0 | Car (incl. vans) | 0.979 | 0.981 | 0.992 | 0.835 |
| 1 | Bus | 0.952 | 0.977 | 0.988 | 0.826 |
| 2 | Truck | 0.887 | 0.916 | 0.935 | 0.722 |
| 3 | Motorcycle | 0.827 | 0.866 | 0.888 | 0.463 |
| **All** | | **0.911** | **0.935** | **0.951** | **0.711** |

> Pedestrian and bicycle classes exist in the weights but are underrepresented, unevaluated, and filtered by default. See the [model card](https://huggingface.co/rfonod/geo-trax) for full details.

To use a different model, point `--model` (CLI) or `extraction.model` (config) to a local `.pt` path or `hf://<org>/<repo>/<file>.pt`; any [Ultralytics](https://github.com/ultralytics/ultralytics)-compatible model works.

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

# Regenerate visualization without re-running extraction
geotrax batch video.mp4 --viz-only --save

# Render specific visualization modes (0: original, 1: stabilized, 2: reference frame, 3: oriented best-fit boxes)
geotrax visualize video.mp4 --save --viz-mode 0 3

# Show lane IDs, hide the speed overlay (requires georeferencing)
geotrax batch video.mp4 --viz-only --save --show-lanes --hide-speed

# Georeference an already-extracted video against orthophotos
geotrax georeference video.mp4 -orf path/to/orthophotos -mf path/to/master_frames

# Aggregated trajectory plots, excluding buses and trucks
geotrax batch path/to/PROCESSED/ --plot-only --plot-aggregate --plot-class-filter 1 2

# Merge multi-drone results for the same locations into a unified dataset
geotrax aggregate path/to/PROCESSED/
```

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

- **video_file_vid_transf.txt** (`<stem><stab_transform_postfix>.txt`): Contains the transformation matrix for each frame in the format:

  ```text
  frame_id, h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

    where:
  - `frame_id`: Frame number of the stabilized frame (starts from `cut_frame_left + 1` since the reference frame itself has no transform).
  - `hij`: Elements of the 3x3 homography matrix that maps each frame (`frame_id`) to the reference frame.

- **video_file.yaml**: Video metadata and the configuration settings used for processing `video_file.mp4`. (This file is saved in the same directory as the input video, not in the output folder.)

- **video_file_mode_X.mp4** (`<stem><visualization_postfix>_mode_<X>.mp4`): Annotated video in four rendering modes (X = 0 / 1 / 2 / 3):
  - **Mode 0**: overlaid on the original (unstabilized) video
  - **Mode 1**: overlaid on the stabilized video
  - **Mode 2**: plotted on the static reference frame
  - **Mode 3**: oriented best-fit boxes on the original video — each box is sized to the vehicle's best-estimated length/width and rotated to its per-frame heading (derived from the camera-motion-free stabilized trajectory and projected back onto the original frame). Requires stabilization to have been run.

  Each version can display vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement. If an input `video_file.csv` file is available in the same directory as the input video, i.e., the converted flight logs, vehicle speed and lane information can also be displayed.

- **video_file.csv** (`<stem><georeferenced_postfix>.csv`): Contains the georeferenced vehicle trajectories in a tabular format. This file includes both geographic and local coordinates, estimated real-world dimensions, kinematic data, road section, and lane information. The columns are:

  ```text
  Vehicle_ID, [Timestamp,] Frame_Number, Ortho_X, Ortho_Y, Local_X, Local_Y, Latitude, Longitude, Vehicle_Length, Vehicle_Width, Vehicle_Class, Vehicle_Speed, Vehicle_Acceleration, Road_Section, Lane_Number, Visibility
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

- **video_file_geo_transf.txt** (`<stem><geo_transform_postfix>.txt`): Contains the 3x3 georeferencing transformation matrix (homography) that maps points from the video's reference frame to the orthomap. The format is a comma-separated list of the 9 matrix elements:

  ```text
  h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

**Note:** *All output files (except `video_file.yaml`) are saved in the configured output folder (default: `results/` sub-folder next to the input video). Trajectory and distribution plots are always written to a `plots/` sub-folder inside the output folder.*

</details>

## Real-World Deployment: The Songdo Experiment

Geo-trax was validated in a large-scale urban traffic monitoring campaign in Songdo, South Korea, where it processed footage from a fleet of 10 drones to produce the [**Songdo Traffic**](https://doi.org/10.5281/zenodo.13828383) dataset. The detection model was trained on the companion [**Songdo Vision**](https://doi.org/10.5281/zenodo.13828407) dataset. Both are described in the [publication](#citation).

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
│       ├── A1.yaml                        # run metadata, saved next to the clip (not in results/)
│       └── results/                       # pipeline outputs, written next to each clip
│           ├── A1.txt                     # pixel-coordinate tracks
│           ├── A1_vid_transf.txt          # stabilization homographies
│           ├── A1_geo_transf.txt          # georeferencing homography
│           ├── A1.csv                     # georeferenced trajectories + kinematics
│           ├── A1_mode_0.mp4              # video with overlaid boxes & trajectories (modes 0/1/2)
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
      month = jun,
      version = {1.0.0},
      doi = {10.5281/zenodo.12119542},
      url = {https://github.com/rfonod/geo-trax},
      license = {MIT}
    }
    ```

## Contributions

Early code received key contributions from [Haechan Cho](https://github.com/cho-96) (georeferencing) and [Sohyeong Kim](https://github.com/shgold) (video/flight-log merging). Community contributions are welcome: open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) for more details.
