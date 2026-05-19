# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/geo-trax)
[![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205)
[![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![Project Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax) [![YouTube](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk)

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is a comprehensive pipeline for extracting high-accuracy georeferenced vehicle trajectories from high-altitude drone imagery. Designed specifically for quasi-stationary aerial monitoring in urban traffic scenarios, Geo-trax transforms raw, bird's-eye view (BEV) video footage into precise, real-world vehicle trajectories. The framework integrates state-of-the-art computer vision and deep learning modules for vehicle detection, tracking, and trajectory stabilization, followed by a georeferencing stage that employs image registration to align the stabilized video frames with an orthophoto. This registration enables the accurate mapping of vehicle trajectories to real-world coordinates. The resulting pipeline supports large-scale traffic studies by delivering spatially and temporally consistent trajectory data suitable for traffic behavior analysis and simulation. Geo-trax is optimized for urban intersections and arterial corridors, where high-fidelity vehicle-level insights are essential for intelligent transportation systems and digital twin applications.

![Geo-trax Visualization GIF](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization.gif?raw=True)

🎬 This animation previews some of the capabilities of Geo-trax. Watch the full demonstration (~4 min) on [YouTube](https://youtu.be/gOGivL9FFLk).

## Pipeline diagram

![Geo-trax pipeline diagram: raw drone video → detection → tracking → stabilization → georeferencing → dataset](assets/geo-trax_pipeline.svg)

🔍 This diagram shows a high-level overview of the Geo-trax processing pipeline — from raw drone footage to georeferenced vehicle trajectories; it also highlights optional human annotation and auxiliary vision-data generation steps.

## Features

1. **Vehicle Detection**: Utilizes a pre-trained YOLO model to detect vehicles (cars, buses, trucks, and motorcycles) in the video frames.
2. **Vehicle Tracking**: Implements a selected tracking algorithm to follow detected vehicles, ensuring robust trajectory data and continuity across frames.
3. **Trajectory Stabilization**: Corrects for unintentional drone movement by aligning trajectories to a reference frame, using bounding boxes of detected vehicles to enhance stability. Leverages the [Stabilo](https://github.com/rfonod/stabilo) 🌀 library, fine-tuned by [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯, to achieve reliable, consistent stabilization.
4. **Georeferencing**: Maps stabilized trajectories to real-world coordinates using an orthophoto and image registration technique.
5. **Dataset Creation**: Compiles trajectory and related metadata (e.g., velocity, acceleration, dimension estimates) into a structured dataset.
6. **Visualization Tools**: Visualizes extracted trajectories, overlays paths on video frames, and generates plots for traffic data analysis.
7. **Auxiliary Tools**: Data wrangling, analysis, and model training scripts/tools provided to support dataset preparation, advanced analytics, and custom model development.
8. **Customization and Configuration**: Flexible configuration options to adjust pipeline settings, including detection/tracking parameters, stabilization methods, and visualization modes.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

### Release Plan

- **Version =1.0.0**
  - Release all data wrangling and analysis tools.
  - Add documentation and examples covering all core functionalities.
  - Host the object detection model file on Hugging Face.

- **Future Versions**
  - Support for custom, user-provided models for detection.
  - Modularization of the pipeline: detection, tracking, and stabilization as separate steps with support for custom reference frames.
  - Rationalized configuration using a single `.yaml` file with nested dictionaries.
  - Support for per-class confidence thresholds and oriented bounding box visualization (with azimuth and dimension estimates).
  - Refactored utilities into focused modules (e.g., `file_utils`, `config_utils`, `data_utils`).
  - Unit tests for all main functions and automated testing via GitHub Actions.
  - Expanded documentation and separate README for analysis tools.
  - Installable package via PyPI (`pip install geo-trax`) with a modular package layout.
  - Upgrade to latest `ultralytics` (>8.2) and `numpy` (>2.0) releases.
  - Batch inference and multi-thread processing for improved scalability.
  - GPU-accelerated image registration and track interpolation in image coordinates.
  - Integration with real-world map visualization tools (e.g., MovingPandas, contextily).

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

1. **Clone or fork the repository**:

    ```bash
    git clone https://github.com/rfonod/geo-trax.git
    cd geo-trax
    ```

2. **Create and activate a Python virtual environment** (Python >= 3.10 and <= 3.12), e.g., using [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install):

    ```bash
    conda create -n geo-trax python=3.11 -y
    conda activate geo-trax
    ```

    or using [venv](https://docs.python.org/3/library/venv.html):

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. **Install dependencies** from `pyproject.toml`:

    ```bash
    pip install -e .
    ```

    > **Note:** Installing in *editable mode* (`-e`) is recommended, as it allows immediate reflection of code changes.

4. **[Optional] Install development dependencies** (for development, testing, or other non-core auxiliary scripts):

    ```bash
    pip install -e '.[dev]'
    ```

## Model Training

The `train/` directory contains scripts for training and exporting custom YOLOv8 detection models using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework, along with a SLURM wrapper for HPC clusters. See [train/README.md](train/README.md) for full usage instructions.

## Batch Processing Example

The `batch_process.py` script can process multiple videos in a directory, including subdirectories, or a single video file.

To view the help message and available options, run:

```bash
python batch_process.py -h
```

Below are some example commands to demonstrate its usage.

#### Example 1: Process all files in a directory without georeferencing

```bash
python batch_process.py path/to/videos/ --no-geo
```

#### Example 2: Customize arguments for a specific video

```bash
python batch_process.py video.mp4 -c cfg/custom_config.yaml
```

#### Example 3: Process and save visualization for a specific video

```bash
python batch_process.py video.mp4 --save
```

#### Example 4: Save tracking results in video without re-running extraction

```bash
python batch_process.py video.mp4 --viz-only --save
```

#### Example 5: Generate plots only for all videos in a directory

```bash
python batch_process.py path/to/videos/ --plot-only --plot
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

    where
  - `frame_id`: Frame number (1, 2, ...).
  - `hij`: Elements of the 3x3 homography matrix that maps each frame (`frame_id`) to the reference frame (frame 0).

- **video.yaml**: Video metadata and the configuration settings used for processing the `video.mp4`. (this file is saved in the same directory as the input video.)

- **video_mode_X.mp4**: Processed video in various visualization modes (X = 0, 1, 2):
  - **Mode 0**: Results overlaid on the original (unstabilized) video.
  - **Mode 1**: Results overlaid on the stabilized video.
  - **Mode 2**: Results plotted on top of the static reference frame.

  Each version can display vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement. If an input `video.csv` file is available in the same directory as the input video, i.e., the converted flight logs, vehicle speed and lane information can be also displayed.

- **video.csv**: Contains the georeferenced vehicle trajectories in a tabular format. This file includes both geographic and local coordinates, estimated real-world dimensions, kinematic data, road section, and lane information. The columns are:

  ```text
  Vehicle_ID, Timestamp, Ortho_X, Ortho_Y, Local_X, Local_Y, Latitude, Longitude, Vehicle_Length, Vehicle_Width, Vehicle_Class, Vehicle_Speed, Vehicle_Acceleration, Road_Section, Lane_Number, Visibility
  ```

    where:
  - `Vehicle_ID`: Unique vehicle identifier.
  - `Timestamp`: Timestamp of the frame (YYYY-MM-DD HH:MM:SS.ms).
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

If you use **Geo-trax** in your research, software, or to generate datasets, please cite the following resources appropriately:

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
      month = may,
      title = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
      url = {https://github.com/rfonod/geo-trax},
      doi = {10.5281/zenodo.12119542},
      version = {0.8.0},
      year = {2026}
    }
    ```

## Contributions

The georeferencing code was developed with contributions from [Haechan Cho](https://github.com/cho-96).

Contributions from the community are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
