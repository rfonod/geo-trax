# Geo-trax: GEO-referenced TRAjectory eXtraction

[![DOI](https://zenodo.org/badge/817002220.svg)](https://zenodo.org/doi/10.5281/zenodo.12119542) ![GitHub Release](https://img.shields.io/github/v/release/rfonod/geo-trax?include_prereleases) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![GitHub](https://img.shields.io/badge/Development-Active-brightgreen)

🚧 **Development Notice** 🚧

> ⚠️ **IMPORTANT:** Geo-trax is currently in its preliminary stages and under active development. Not all features are complete, and significant changes may occur. It is recommended for experimental use only. Please report any issues you encounter, and feel free to contribute to the project.

Geo-trax is an end-to-end pipeline designed for extracting high-accuracy georeferenced vehicle trajectories from drone imagery. It is particularly tailored for scenarios where the drone remains quasi-stationary, such as monitoring an intersection. The pipeline is optimized for videos captured from a bird's-eye view (top-down) perspective and includes the following key steps:

1. **Vehicle Detection** (✅): Utilizes a pre-trained YOLOv8 model to detect vehicles (cars, buses, trucks, and motorcycles) in the video frames.
2. **Vehicle Tracking** (✅): Employs a selected tracking algorithm to track the detected vehicles in the video frames.
3. **Trajectory Stabilization** (✅): Maps each vehicle's trajectory to the reference (first) frame of the video, ensuring consistency despite any unintentional drone movements. Uses detected vehicle bounding boxes to enhance the stabilization process. Leverages the [stabilo package](https://github.com/rfonod/stabilo) for this purpose.
4. **Georeferencing** (👷🏼): Leverages an orthomap and the reference frame to convert the stabilized trajectories into georeferenced data, mapping the vehicle paths in real-world coordinates.
5. **Dataset Creation** (👷🏼): Compiles trajectories and associated metadata (e.g., vehicle velocity, acceleration, and dimension estimates), into a comprehensive dataset.
6. **Visualization Tools** (👷🏼): Generates visualizations of the extracted results, including the vehicle trajectories overlaid on the video frames or various plots for traffic data analysis.

This is a preliminary version of the pipeline. Future versions will include more detailed documentation, a more user-friendly interface, and additional functionalities.

<details>
<summary><b>🚀 Planned Enhancements and Future Features</b></summary>

### Release Plan
- **Version 1.0**
    - Full implementation of georeferencing (Point 4 above).
    - Comprehensive dataset creation with all metadata (Point 5 above).
    - Release of visualization tools (Point 6 above).

- **Feature Roadmap**
    - Upgrades to the latest ultralytics (>8.2) and numpy (>2.0) versions.
    - Support for additional tracking algorithms and vehicle types.
    - Release of the (re-)training tools for the detection model.
    - Tools to compare the extracted trajectories against on-board sensors.
    - More detailed and accessible documentation on using the pipeline with custom data.

- **Technical Enhancements**
    - Switch to a package layout for improved modularity.
    - Implement batch inference for detection and/or multi-thread processing for scalability.
    - Implement GPU-accelerated image registration and track interpolation in image coordinates.
    - Data wrangling tools to pre-process raw video data for the pipeline.
    - Release the data wrangling tools to pre-process raw video data for the pipeline.

- **Documentation and Compliance**
    - Write comprehensive documentation with more example use cases (`docs` folder).
    - Maintain a list of known limitations, such as potential discrepancies in ffmpeg backend versions of OpenCV on different machines.
    - Add unit tests for key functionalities.

- **Development Infrastructure**
    - Create GitHub Actions for automated testing.
    - Publish the package on PyPI.

</details>

## Installation

1. **Create a Python Virtual Environment** (Python >= 3.9) using e.g., [Miniconda3](https://docs.anaconda.com/free/miniconda/):
    ```bash
    conda create -n geo-trax python=3.9 -y
    conda activate geo-trax
    ```

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/rfonod/geo-trax.git
    cd geo-trax
    ```

3. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Batch Processing Usage Examples

The `batch_process.py` script can process multiple videos in a directory, including subdirectories, or a single video file.

**Example 1: Process a Directory Without Saving Output Videos**
```bash
python batch_process.py path/to/videos/
```

**Example 2: Customize Arguments for a Specific Video**
```bash
python batch_process.py video.mp4 -c cfg/custom_config.yaml
```

**Example 3: Process and Save Visualization of a Specific Video**
```bash
python batch_process.py video.mp4 --save
```

**Example 4: Save Tracking Results in Video Without Re-running Tracking**
```bash
python batch_process.py video.mp4 --viz-only --save
```

**Example 5: Overwrite Existing Files with No Confirmation**
```bash
python batch_process.py path/to/videos/ -o -y
```

<details>
<summary><b>📁 Output File Formats</b></summary>

- **video.txt**: Contains the extracted vehicle trajectories in the format:
    ```
    frame_id, vehicle_id, x_c(unstab), y_c(unstab), w(unstab), h(unstab), x_c(stab), y_c(stab), w(stab), h(stab), class_id, confidence, vehicle_length, vehicle_width
    ```
    where:
    - `frame_id`: Frame number.
    - `vehicle_id`: Unique vehicle identifier.
    - `x_c(unstab)`, `y_c(unstab)`: Unstabilized vehicle centroid coordinates.
    - `w(unstab)`, `h(unstab)`: Unstabilized vehicle bounding box width and height.
    - `x_c(stab)`, `y_c(stab)`: Stabilized vehicle centroid coordinates.
    - `w(stab)`, `h(stab)`: Stabilized vehicle bounding box width and height.
    - `class_id`: Vehicle class identifier (0: car (incl. vans), 1: bus, 2: truck, 3: motorcycle)
    - `confidence`: Detection confidence score.
    - `vehicle_length`, `vehicle_width`: Estimated vehicle dimensions in pixels.

- **video_vid_transf.txt**: Contains the transformation matrix for each frame in the format:
    ```
    frame_id, h11, h12, h13, h21, h22, h23, h31, h32, h33
    ```
    where
    - `frame_id`: Frame number (>0).
    - `hij`: Elements of the 3x3 homography matrix that maps each frame (`frame_id`) to the reference frame (frame 0).

- **video.yaml**: Video metadata and the configuration settings used for processing. Saved alongside the input video.

- **video_mode_X.mp4**: Processed video in various visualization modes (X = 0, 1, 2):
  - **Mode 0**: Results overlaid on the original (unstabilized) video.
  - **Mode 1**: Results overlaid on the stabilized video.
  - **Mode 2**: Results plotted on top of a static reference frame image.

  Each version displays vehicle bounding boxes, IDs, class labels, confidence scores, and short trajectory trails that fade and vary in thickness to indicate the recency of the movement.

- **video.csv**: Contains the georeferenced vehicle trajectories in a tabular format with additional metadata (TBD).

- **video_geo_transf.txt**: Georeferencing transformation matrix between the reference frame and the orthomap (TBD).

**Note:** *All output files (except `video.yaml`) are saved in the `results` folder relative to the input video.*

</details>

## Citing This Work

If you use this project in your academic research, commercial products, or any published material, please acknowledge its use by citing it. For the correct citation, refer to the DOI badge above, which links to the appropriate version of the release on Zenodo. Ensure you reference the version you actually used. A formatted citation can also be obtained from the top right of the [GitHub repository](https://github.com/geo-trax).

```bibtex
@software{Fonod_Geo-trax_2024,
author = {Fonod, Robert},
license = {MIT},
month = jun,
title = {{Geo-trax}},
url = {https://github.com/rfonod/geo-trax},
doi = {10.5281/zenodo.12119542},
version = {0.1.0},
year = {2024}
}
``` 

## Note on Upcoming Publication

An academic paper is currently being prepared based on this work. Once published, this section will be updated with a specific reference to the paper, which will then be the preferred citation for this project. Please keep an eye on this space for updates.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/geo-trax/issues) or submit a pull request. Your contributions are greatly appreciated!

## License

This project is licensed under the MIT License, an [OSI-approved](https://opensource.org/licenses/MIT) open-source license, which allows for both academic and commercial use. By citing this project, you help support its development and acknowledge the effort that went into creating it. For more details, see the [LICENSE](LICENSE) file. Thank you!