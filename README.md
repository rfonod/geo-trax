# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/817002220.svg)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg?style=flat)](https://arxiv.org/abs/2411.02136) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/geo-trax)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advanced-computer-vision-for-extracting/object-detection-on-songdo-vision)](https://paperswithcode.com/sota/object-detection-on-songdo-vision?p=advanced-computer-vision-for-extracting)

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is a comprehensive pipeline for extracting high-accuracy georeferenced vehicle trajectories from high-altitude drone imagery. Designed specifically for quasi-stationary aerial monitoring in urban traffic scenarios, Geo-trax transforms raw, bird’s-eye view video footage into precise, real-world vehicle trajectories. The framework integrates state-of-the-art computer vision and deep learning modules for vehicle detection, tracking, and trajectory stabilization, followed by a georeferencing stage that employs image registration to align the stabilized video frames with an orthophoto. This registration enables the accurate mapping of vehicle trajectories to real-world coordinates. The resulting pipeline supports large-scale traffic studies by delivering spatially and temporally consistent trajectory data suitable for traffic behavior analysis and simulation. Geo-trax is optimized for urban intersections and arterial corridors, where high-fidelity vehicle-level insights are essential for intelligent transportation systems (ITS) and digital twin applications.

## Features

1. **Vehicle Detection**: Utilizes a pre-trained YOLOv8 model to detect vehicles (cars, buses, trucks, and motorcycles) in the video frames.
2. **Vehicle Tracking**: Implements a selected tracking algorithm to follow detected vehicles, ensuring robust trajectory data and continuity across frames.
3. **Trajectory Stabilization**: Corrects for unintentional drone movement by aligning trajectories to a reference frame, using bounding boxes of detected vehicles to enhance stability. Leverages the [stabilo](https://github.com/rfonod/stabilo) 🚀 library, fine-tuned by [stabilo-optimize](https://github.com/rfonod/stabilo-optimize), to achieve reliable, consistent stabilization.
4. **Georeferencing**: Maps stabilized trajectories to real-world coordinates using an orthophoto and image registration technique.
5. **Dataset Creation**: Compiles trajectory and related metadata (e.g., velocity, acceleration, dimension estimates) into a structured dataset.
6. **Visualization Tools**: Visualizes extracted trajectories, overlays paths on video frames, and generates plots for traffic data analysis.
7. **Auxiliary Tools (👷🏼)**: Data wrangling, analysis, and model training scripts/tools will be provided in future releases to support dataset preparation, advanced analytics, and custom model development.
8. **Customization and Configuration**: Flexible configuration options to adjust pipeline settings, including detection/tracking parameters, stabilization methods, and visualization modes.

> **Note:** This is a preliminary version of the pipeline. Some functionalities, especially auxiliary tools for data wrangling, analysis, and model training, are under development (👷🏼) and will be included in future releases.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

### Release Plan

- **Version >1.0.0**
  - Tools for comparing extracted trajectories with on-board sensor data.
  - Release auxiliary tools for data wrangling, analysis, and (re-)training the detection model.
  - Basic documentation and examples covering all core functionalities.

- **Version >1.0.0**
  - Pre-processing tools for raw video input.
  - Expanded documentation, tutorials (docs folder), and sample examples.
  - List of known limitations, e.g., ffmpeg backend version discrepancies in OpenCV.
  - Comprehensive unit tests for critical functions and end-to-end tests for the entire pipeline.
  - Publishing on PyPI for simplified installation and distribution.

- **Version 2.0.0**
  - Upgrades to the latest ultralytics (>8.2) and numpy (>2.0) versions.
  - Support for additional tracking algorithms and broader vehicle type recognition.
  - Transition to a modular package layout for enhanced maintainability.
  - Implementation of batch inference and multi-thread processing to improve scalability.
  - Automated testing workflows with GitHub Actions.

</details>

## Field Deployment

Geo-trax was validated in a large-scale urban traffic monitoring experiment conducted in Songdo, South Korea. In this study, Geo-trax was used to process aerial video data captured by a fleet of 10 drones, resulting in the creation of the [**Songdo Traffic** dataset](https://doi.org/10.5281/zenodo.13828383). The underlying vehicle detection model in Geo-trax was trained using the [**Songdo Vision** dataset](https://doi.org/10.5281/zenodo.13828407). Both datasets are described in detail in the associated publication (see [Citation](#citation) below).

## Installation

1. **Create and activate a Python virtual environment** (Python ≥ 3.9), e.g., using [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install):

    ```bash
    conda create -n geo-trax python=3.11 -y
    conda activate geo-trax
    ```

2. **Clone or fork the repository**:

    ```bash
    git clone https://github.com/rfonod/geo-trax.git
    cd geo-trax
    ```

3. **Install dependencies** using `pyproject.toml`:

    ```bash
    pip install -e .
    ```

    > **Note:** Installing in *editable mode* (`-e`) is recommended, as it allows immediate reflection of code changes.

4. **[Optional] Install development dependencies** (for development, testing, or other non-core auxiliary scripts):

    - In `bash`:

      ```bash
      pip install -e .[dev]
      ```

    - In `zsh` (quotes required to prevent shell expansion of `[]`):

      ```zsh
      pip install -e '.[dev]'
      ```

## Batch Processing Example

The `batch_process.py` script can process multiple videos in a directory, including subdirectories, or a single video file.

#### Example 1: Process all files in a directory

```bash
python batch_process.py path/to/videos/
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

#### Example 5: Overwrite existing files with no confirmation prompt

```bash
python batch_process.py path/to/videos/ -o -y
```

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

  Each version can display vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement. If `video.csv` is available, vehicle speed and lane information can be also displayed.

- **video.csv**: Contains the georeferenced vehicle trajectories in a tabular format with additional metadata (TBD).

- **video_geo_transf.txt**: Georeferencing transformation matrix between the reference frame and the orthomap (TBD).

**Note:** *All output files (except `video.yaml`) are saved in the `results` folder relative to the input video.*

</details>

## Citation

If you use Geo-trax or the associated datasets in your research or applications, please cite the following resources appropriately:

1. **Preferred Citation:** For academic or research-related use, please cite the associated paper. A preprint is currently available on [arXiv](https://arxiv.org/abs/2411.02136), with formal publication forthcoming:

    ```bibtex
    @misc{fonod2025advanced,
      title={Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery}, 
      author={Robert Fonod and Haechan Cho and Hwasoo Yeo and Nikolas Geroliminis},
      year={2025},
      eprint={2411.02136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02136},
      doi={https://doi.org/10.48550/arXiv.2411.02136}
    }
    ```

2. **Repository Citation:** If you reference or build upon the Geo-trax software framework itself, please cite the corresponding Zenodo release:

    ```bibtex
    @software{fonod2025geo-trax,
      author = {Fonod, Robert},
      license = {MIT},
      month = may,
      title = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
      url = {https://github.com/rfonod/geo-trax},
      doi = {10.5281/zenodo.12119542},
      version = {0.5.0},
      year = {2025}
    }
    ```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
