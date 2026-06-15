# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![PyPI - Version](https://img.shields.io/pypi/v/geo-trax)](https://pypi.org/project/geo-trax/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/geo-trax?label=total%20downloads)](https://pepy.tech/project/geo-trax) [![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/geo-trax?color=%234c1)](https://pypi.org/project/geo-trax/) [![CI](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/geo-trax/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.9--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/geo-trax) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/issues) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![Project Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax) [![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk)

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is a comprehensive pipeline for extracting high-accuracy georeferenced vehicle trajectories from high-altitude drone imagery. Designed specifically for quasi-stationary aerial monitoring in urban traffic scenarios, Geo-trax transforms raw, bird's-eye view (BEV) video footage into precise, real-world vehicle trajectories. The framework integrates state-of-the-art computer vision and deep learning modules for vehicle detection, tracking, and trajectory stabilization, followed by a georeferencing stage that employs image registration to align the stabilized video frames with an orthophoto. This registration enables the accurate mapping of vehicle trajectories to real-world coordinates. The resulting pipeline supports large-scale traffic studies by delivering spatially and temporally consistent trajectory data suitable for traffic behavior analysis and simulation. Geo-trax is optimized for urban intersections and arterial corridors, where high-fidelity vehicle-level insights are essential for intelligent transportation systems and digital twin applications.

![Geo-trax Visualization GIF](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization.gif?raw=True)

🎬 This animation previews some of the capabilities of Geo-trax. Watch the full demonstration (~4 min) on [YouTube](https://youtu.be/gOGivL9FFLk).

## Pipeline diagram

![Geo-trax pipeline diagram: raw drone video → detection → tracking → stabilization → georeferencing → dataset](assets/geo-trax_pipeline.svg)

🔍 The core pipeline (solid box) transforms raw drone video into stabilized, pixel-coordinate vehicle trajectories. Optional extensions include: georeferencing to real-world coordinates via orthophoto image registration; vision dataset creation through (pre-)annotation of sampled frames for custom detector training; and bundled visualization, analysis, and probe vehicle validation tools, applicable to both pixel-coordinate and georeferenced outputs.

## Features

1. **Vehicle Detection**: Utilizes a pre-trained YOLO model to detect vehicles (cars, buses, trucks, and motorcycles) in the video frames.
2. **Vehicle Tracking**: Implements the selected tracking algorithm to follow detected vehicles, ensuring robust trajectory data and continuity across frames.
3. **Trajectory Stabilization**: Corrects for unintentional drone movement by aligning trajectories to a reference frame, using bounding boxes of detected vehicles to enhance stability. Leverages the [Stabilo](https://github.com/rfonod/stabilo) 🌀 library, fine-tuned by [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯, to achieve reliable, consistent stabilization.
4. **Georeferencing**: Maps stabilized trajectories to real-world coordinates using an orthophoto and an image registration technique.
5. **Dataset Creation**: Compiles trajectory and related metadata (e.g., velocity, acceleration, dimension estimates) into a structured dataset.
6. **Visualization Tools**: Visualizes extracted trajectories, overlays paths on video frames, and generates plots for traffic data analysis.
7. **Auxiliary Tools**: Provides data wrangling, analysis, and model training scripts/tools to support dataset preparation, advanced analytics, and custom model development.
8. **Customization and Configuration**: Flexible configuration options to adjust pipeline settings, including detection/tracking parameters, stabilization methods, and visualization modes.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- Comprehensive documentation in a dedicated `docs/` folder, including tool-specific READMEs.
- Modularized, OOP-based pipeline with custom reference frame support and georeferencing leveraging Stabilo's image-matching backend.
- Rationalized single-file YAML configuration.
- Per-class confidence thresholds and oriented bounding box visualization (using azimuth and dimension estimates).
- Trajectory interpolation and SAHI-based small-object detection.
- Batch inference, GPU-accelerated image registration, and multi-thread processing.
- Real-world map visualization (e.g., MovingPandas, contextily) and interactive web app.

</details>

<details>
<summary><b>🔗 Related Projects</b></summary>

Geo-trax integrates with and complements several specialized tools:

- **[Stabilo](https://github.com/rfonod/stabilo) 🌀** — Python library for video and trajectory stabilization using robust homography transformations. Supports various feature detectors, RANSAC algorithms, and user-defined masks. Used as Geo-trax's core stabilization engine.

- **[Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯** — Benchmarking and hyperparameter optimization framework for Stabilo. Evaluates stabilization performance through ground truth-free assessment using random perturbations. Used to fine-tune Geo-trax stabilization parameters.

- **[HBB2OBB](https://github.com/rfonod/hbb2obb) 📦** — Converts horizontal bounding boxes to oriented bounding boxes using SAM segmentation models. Can enhance Geo-trax outputs when object orientation is needed for downstream analysis.

</details>

## Field Deployment

Geo-trax was validated in a large-scale urban traffic monitoring experiment conducted in Songdo, South Korea. In this study, Geo-trax was used to process aerial video data captured by a fleet of 10 drones, resulting in the creation of the [**Songdo Traffic**](https://doi.org/10.5281/zenodo.13828383) dataset. The underlying vehicle detection model in Geo-trax was trained using the [**Songdo Vision**](https://doi.org/10.5281/zenodo.13828407) dataset. Both datasets are described in detail in the associated publication, see the [citation](#citation) section below.

🎥 *Demo video of Geo-trax applied to the Songdo field experiment:* [https://youtu.be/gOGivL9FFLk](https://youtu.be/gOGivL9FFLk)

## Installation

It is recommended to create and activate a **Python virtual environment** (Python >= 3.9 and <= 3.13) first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

<details>
<summary>Alternatives: conda or uv</summary>

**[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):**
```bash
conda create -n geo-trax python=3.11 -y
conda activate geo-trax
```

**[uv](https://docs.astral.sh/uv/getting-started/installation/) (fastest; use `uv pip install` in the options below):**
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
</details>

Then, install geo-trax using one of the following options:

### Option 1: Install from PyPI

```bash
python -m pip install geo-trax
```

This installs the `geotrax` command-line interface (see [Batch Processing Example](#batch-processing-example)) together with the bundled configuration tree (`geotrax/cfg/`).

> [!NOTE]
> The default detection model (`models/yolov8s_merger8_exp1.pt`) is distributed with the repository, not the PyPI package. Download the [`models/`](models/) folder (weights + class-names YAML) into your working directory, or point the `model:` key of a custom Ultralytics config to its location.

### Option 2: Install from Local Source

Recommended for development or model training — the default detection model weights ship with the repository. Clone (or fork) the repository and install the package from the local source:

```bash
git clone --depth 1 https://github.com/rfonod/geo-trax.git   # latest snapshot only; drop --depth 1 for full history
cd geo-trax
python -m pip install -e .   # pip  (-e: editable mode, reflects code changes without reinstalling)
# uv pip install -e .        # uv   (faster; requires uv venv from above)
# poetry install             # Poetry (auto-manages virtualenv; skip the virtual environment step)
```

**[Optional] Install development dependencies** (for development, testing, or other non-core auxiliary scripts):

```bash
python -m pip install -e '.[dev]'   # pip
# uv pip install -e '.[dev]'        # uv
# poetry install --extras dev       # Poetry
```

## Configuration

The entire pipeline is driven by a **single, self-contained pipeline config file**. Every setting — detection, tracking, stabilization, georeferencing, visualization, and plotting — lives in one YAML file; there are no linked sub-config files. The defaults ship bundled inside the installed package, so there is nothing to download. Four ready-made presets are bundled and can be selected by name with `-c <name>` (e.g. `geotrax extract video.mp4 -c confident`): `default` (balanced baseline), `confident` (higher confidence, fewer false positives), `lenient` (lower confidence, higher recall), and `stable` (full-resolution stabilization).

<details>
<summary><b>⚙️ Inspecting, selecting, and customizing the pipeline config</b></summary>

Manage the bundled configs with the `geotrax config` command:

```bash
geotrax config show              # show where the bundled configs live and list the presets
geotrax config show default      # print a preset's full contents to the terminal
geotrax config copy              # copy the presets into the current directory as <name>_copy.yaml
geotrax config copy -o ~/myproj  # copy into a specific directory
```

**Selecting a preset.** Pass a bundled preset name straight to `-c` — no file path needed:

```bash
geotrax extract video.mp4 -c confident   # uses the bundled 'confident' preset
geotrax batch  video.mp4 -c stable       # uses the bundled 'stable' preset
```

`-c` is resolved as given (absolute or relative to the working directory) and then against the bundled config directory, so a bare preset name like `confident` works from anywhere, while a path like `./my_config.yaml` points to your own file.

**Customizing.** Copy a preset and pass your edited copy to any command with `-c`:

```bash
geotrax config copy
# edit default_copy.yaml ...
geotrax extract video.mp4 -c default_copy.yaml
```

To switch the tracking algorithm, set `tracker.active` in the config (see [Tracking Algorithms](#tracking-algorithms)).

</details>

## Model Training

The `train/` directory contains scripts for training and exporting custom YOLOv8 detection models using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework, along with a SLURM wrapper for HPC clusters. See [train/README.md](train/README.md) for full usage instructions.

## Tracking Algorithms

Geo-trax supports six multi-object trackers bundled with [Ultralytics](https://github.com/ultralytics/ultralytics) (`>=8.4.63`): **BoT-SORT** (default), **ByteTrack**, **OC-SORT**, **Deep OC-SORT**, **FastTracker**, and **TrackTrack**. The full parameter block for all six is inlined in the `tracker:` section of the pipeline config; selection is purely config-driven — set `tracker.active` to the desired name. No code changes are needed.

> 💡 Run `geotrax config show default` to print the bundled config and inspect the `tracker:` section (all six blocks, every parameter documented inline).

<details>
<summary><b>🚗 Tracker comparison and selection guidance</b></summary>

| Tracker | `tracker.active` | ReID | GMC¹ | Pros | Cons |
|---------|------------------|:----:|:----:|------|------|
| **BoT-SORT** (default) | `botsort` | optional | ✅ | Strong all-round accuracy; motion-based with optional appearance (ReID) and in-tracker camera-motion compensation. | Slower than ByteTrack; ReID adds compute and weights. |
| **ByteTrack** | `bytetrack` | ❌ | ❌ | Fastest and simplest; two-stage association recovers low-confidence detections. | No appearance or motion compensation → more ID switches under occlusion or camera motion. |
| **OC-SORT** | `ocsort` | ❌ | ❌ | Observation-centric motion model; robust to non-linear motion and brief occlusions; lightweight, ReID-free. | No appearance cues; weaker on long occlusions and dense, visually similar targets. |
| **Deep OC-SORT** | `deepocsort` | optional | optional | OC-SORT plus appearance embeddings and optional GMC; fewer ID switches in crowded scenes. | Heaviest OC-SORT variant when ReID is enabled; more tuning. |
| **FastTracker** | `fasttrack` | ❌ | ❌ | Occlusion-aware ByteTrack variant with Kalman rollback and init-IoU suppression; good ID retention through brief occlusions at low cost. | Newer/less battle-tested; ReID-free; several occlusion knobs to tune. |
| **TrackTrack** | `tracktrack` | optional | ✅ | Multi-cue cost (HMIoU + ReID + confidence + angle) with iterative assignment and track-aware initialisation. | Most parameters to tune; higher compute. |

> ¹ **GMC** (global motion compensation) corrects for camera/drone movement *inside the tracker's association step*. It runs during tracking and is independent of Geo-trax's separate [Stabilo](https://github.com/rfonod/stabilo) stage, which stabilizes the already-extracted trajectories against a reference frame as post-processing and has no effect on tracking itself.

**Guidance for drone (BEV) footage:** start with **BoT-SORT** (the default). If throughput matters more than identity consistency, try **ByteTrack** or **OC-SORT**. For dense scenes with frequent occlusions, **Deep OC-SORT** or **TrackTrack** (with ReID enabled) tend to retain identities best, at the cost of speed. To compare two or more trackers on your own data, see [`tools/compare_tracking.py`](tools/compare_tracking.py).

</details>

All tracker parameters (confidence thresholds, track buffer, matching thresholds, ReID, GMC, etc.) live in the `tracker:` section of the pipeline config alongside every other pipeline setting, with one block per tracker. Run `geotrax config copy` to get an editable local copy. For full algorithm details, see the [Ultralytics tracking docs](https://docs.ultralytics.com/modes/track/).

## Batch Processing Example

Installing geo-trax provides a single `geotrax` command with one subcommand per pipeline stage: `batch` (primary entry point), `extract`, `georeference`, `visualize`, `plot`, `aggregate`, and `config` (config management). Run `geotrax -h` for the overview and `geotrax <command> -h` for the per-command option reference (`python -m geotrax …` works identically).

The `geotrax batch` command can process multiple videos in a directory, including subdirectories, or a single video file.

To view the help message and available options, run:

```bash
geotrax batch -h
```

Below are some example commands to demonstrate its usage.

#### Example 1: Process all files in a directory without georeferencing

```bash
geotrax batch path/to/videos/ --no-geo
```

#### Example 2: Customize arguments for a specific video

```bash
geotrax batch video.mp4 -c path/to/custom_config.yaml
```

#### Example 3: Save tracking results in a video without re-running extraction

```bash
geotrax batch video.mp4 --viz-only --save
```

#### Example 4: Show lane IDs and hide speeds in the visualization (requires georeferencing)

```bash
geotrax batch video.mp4 --viz-only --save --show-lanes --hide-speeds
```

#### Example 5: Generate aggregated trajectory plots only from existing results, excluding buses and trucks

```bash
geotrax batch path/to/processed-data/ --plot-only --plot-aggregate --plot-class-filter 1 2
```

> [!TIP]
> See [data/README.md](data/README.md) for sample data and testing examples.

<details>
<summary><b>📁 Output File Formats</b></summary>
Suppose the input video is named `video.mp4`. The output files will be saved in the `results` folder relative to the input video. The following files will be generated:

- **video.txt**: Contains the extracted vehicle trajectories in the following format:

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

- **video_vid_transf.txt**: Contains the transformation matrix for each frame in the format:

  ```text
  frame_id, h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

    where:
  - `frame_id`: Frame number of the stabilized frame (starts from `cut_frame_left + 1` since the reference frame itself has no transform).
  - `hij`: Elements of the 3x3 homography matrix that maps each frame (`frame_id`) to the reference frame.

- **video.yaml**: Video metadata and the configuration settings used for processing the `video.mp4`. (This file is saved in the same directory as the input video.)

- **video_mode_X.mp4**: Processed video in various visualization modes (X = 0, 1, 2):
  - **Mode 0**: Results overlaid on the original (unstabilized) video.
  - **Mode 1**: Results overlaid on the stabilized video.
  - **Mode 2**: Results plotted on top of the static reference frame.

  Each version can display vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement. If an input `video.csv` file is available in the same directory as the input video, i.e., the converted flight logs, vehicle speed and lane information can also be displayed.

- **video.csv**: Contains the georeferenced vehicle trajectories in a tabular format. This file includes both geographic and local coordinates, estimated real-world dimensions, kinematic data, road section, and lane information. The columns are:

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

- **video_geo_transf.txt**: Contains the 3x3 georeferencing transformation matrix (homography) that maps points from the video's reference frame to the orthomap. The format is a comma-separated list of the 9 matrix elements:

  ```text
  h11, h12, h13, h21, h22, h23, h31, h32, h33
  ```

**Note:** *All output files (except `video.yaml`) are saved in the `results` folder relative to the input video.*

</details>

## Citation

If you use **Geo-trax** in your research, software, or dataset generation, please cite the following resources appropriately:

1. **Preferred Citation:** Please cite the associated article for any use of the Geo-trax framework, including research, applications, and derivative work:

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

2. **Repository Citation:** If you reference, modify, or build upon the Geo-trax software itself, please also cite the corresponding Zenodo release:

    ```bibtex
    @software{fonod2026geo-trax,
      author = {Fonod, Robert},
      license = {MIT},
      month = jun,
      title = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
      url = {https://github.com/rfonod/geo-trax},
      doi = {10.5281/zenodo.12119542},
      version = {1.0.0},
      year = {2026}
    }
    ```

## Contributions

The georeferencing code was developed with contributions from [Haechan Cho](https://github.com/cho-96).

Contributions from the community are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
