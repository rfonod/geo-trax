# Geo-trax

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases)](https://github.com/rfonod/geo-trax/releases) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/geo-trax)](https://github.com/rfonod/geo-trax/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/817002220.svg)](https://zenodo.org/doi/10.5281/zenodo.12119542) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg?style=flat)](https://arxiv.org/abs/2411.02136) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/geo-trax)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advanced-computer-vision-for-extracting/object-detection-on-songdo-vision)](https://paperswithcode.com/sota/object-detection-on-songdo-vision?p=advanced-computer-vision-for-extracting)

🚧 **Development Notice** 🚧

> ⚠️ **IMPORTANT:** Geo-trax is currently in its preliminary stages and under active development. Not all features are complete, and significant changes may occur. It is recommended for experimental use only. Please report any issues you encounter, and feel free to contribute to the project.

**Geo-trax** (GEO-referenced TRAjectory eXtraction) is an end-to-end pipeline for extracting high-accuracy, georeferenced vehicle trajectories from high-altitude, bird’s-eye view drone footage, addressing critical challenges in urban traffic analysis and real-world georeferencing. Designed for quasi-stationary drone monitoring of intersections or road segments, Geo-trax utilizes advanced computer vision and deep learning methods to deliver high-quality data and support scalable, precise traffic studies. This pipeline transforms raw, top-down video data into georeferenced vehicle trajectories in real-world coordinates, enabling detailed analysis of vehicle dynamics and behavior in dense urban environments.

## Features

1. **Vehicle Detection** (✅): Utilizes a pre-trained YOLOv8 model to detect vehicles (cars, buses, trucks, and motorcycles) in the video frames.
2. **Vehicle Tracking** (✅): Implements a selected tracking algorithm to follow detected vehicles, ensuring robust trajectory data and continuity across frames.
3. **Trajectory Stabilization** (✅): Corrects for unintentional drone movement by aligning trajectories to a reference frame, using bounding boxes of detected vehicles to enhance stability. Leverages the [stabilo](https://github.com/rfonod/stabilo) 🚀 library, fine-tuned by [stabilo-optimize](https://github.com/rfonod/stabilo-optimize), to achieve reliable, consistent stabilization.
4. **Georeferencing** (✅): Maps stabilized trajectories to real-world coordinates using an orthophoto and image registration technique.
5. **Dataset Creation** (✅): Compiles trajectory and related metadata (e.g., velocity, acceleration, dimension estimates) into a structured dataset.
6. **Visualization Tools** (✅/👷🏼): Provides tools to visualize the extracted trajectories, overlaying paths on video frames and generating various plots for traffic data analysis.
7. **Customization and Configuration** (✅): Offers flexible configuration options to adjust the pipeline settings, including detection and tracking parameters, stabilization methods, and visualization modes.

This is a preliminary version of the pipeline with some functionalities not being implemented (👷🏼). Future releases will include more detailed documentation, a more user-friendly interface, and additional functionalities.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

### Release Plan

- **Version >1.0.0**
  - Complete georeferencing functionality (Point 4 above).
    - Comprehensive dataset creation with all metadata (Point 5 above).
    - Visualization and plotting tools (Point 6 above).
    - Tools for comparing extracted trajectories with on-board sensor data.
    - Basic documentation and examples covering all core functionalities.

- **Version >1.0.0**
  - Release tools for (re-)training the detection model.
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

## Installation

1. **Create and activate a Python virtual environment** (Python >= 3.9), e.g., using [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install):

    ```bash
    conda create -n geo-trax python=3.11 -y
    conda activate geo-trax
    ```

2. **Clone or fork the repository**:

    ```bash
    git clone https://github.com/rfonod/geo-trax.git
    cd geo-trax
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **[Optional] Install development dependencies** (for development, testing, or model training):

    ```bash
    pip install -r requirements-dev.txt
    ```

  > **Tip:** Only install these if you plan to contribute code, run tests, or retrain models.

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

## Citing This Work

If you use this project in your academic research, commercial products, or any published material, please acknowledge its use by citing it.

1. **Preferred Citation:** For research-related references, please cite the related paper once it is formally published. A preprint is currently available on [arXiv](https://arxiv.org/abs/2411.02136):

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

2. **Repository Citation:** For direct use of the geo-trax framework, please cite the software release version on Zenodo. You may refer to the DOI badge above for the correct version or use the BibTeX below:

    ```bibtex
    @software{fonod2025geo-trax,
      author = {Fonod, Robert},
      license = {MIT},
      month = apr,
      title = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
      url = {https://github.com/rfonod/geo-trax},
      doi = {10.5281/zenodo.12119542},
      version = {0.4.0},
      year = {2025}
    }
    ```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
