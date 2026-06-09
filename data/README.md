# Data Directory

This directory contains a short drone video clip, `U_video_cut.mp4` (5 seconds), and its associated flight log, `U_video_cut.csv`. These files alone enable quick testing and demonstration of trajectory extraction in pixel coordinates.

The video clip is sourced from the [Songdo Traffic](https://doi.org/10.5281/zenodo.13828383) dataset. It was created by extracting the first 5 seconds from `U_D10_2022-10-07_PM5_60s.mp4` using the following command:

```bash
python tools/recut_video_and_csv.py data/sample_videos/U_D10_2022-10-07_PM5_60s.mp4 -s 0 -e 150 -o data/U_video_cut.mp4 -ec
```

## Reproducing Results

### Pixel-Coordinate Results

To reproduce the pixel-coordinate results in the `data/results-pixel/` directory, run the following command from the repository root:

```bash
python batch_process.py data/U_video_cut.mp4 --no-geo --show-class-names --show-conf
```

### Full Pipeline Results

To reproduce the full Geo-trax pipeline results in the `data/results-full/` directory, including georeferencing, road segmentation, kinematics, and real-world vehicle dimension estimation, run the following command from the repository root:

```bash
python batch_process.py data/U_video_cut.mp4 -of data/orthophotos -osf data/segmentations -mf data/master_frames --show-lanes --segmentations
```

> **Note:** The trajectory and distribution plots generated from this 5-second sample are not statistically meaningful due to the limited sample size. Longer video clips are needed for representative results.

**Prerequisites:** Download the required files as described in the [Sample Videos and Data](#sample-videos-and-data-for-full-pipeline-testing) section:

- Orthophoto files: `orthophotos/U_center.txt`, `orthophotos/U.png`, `orthophotos/ortho_parameters.txt`
- Segmentation files: `segmentations/U.csv`; optional overlay PNG can be generated later (see [overlay instructions](#generate-segmentation-overlays)).
- Master frame files: `master_frames/U.png`, `master_frames/U.txt`

> **Tip:** You can easily process any sample video from the Songdo Traffic dataset by replacing `U_video_cut.mp4` with the desired filename (e.g., `A_D1_2022-10-07_PM5_60s.mp4`).

## Sample Videos and Data for Full Pipeline Testing

The [Songdo Experiment](../README.md#field-deployment) provides ready-to-use files to test the full capabilities of the Geo-trax pipeline, including sample BEV drone footage from 20 different intersections (60-second clips), orthophotos and road segmentation masks for each intersection, and optional master frame files for consistent georeferencing.

All files are available from [Zenodo](https://doi.org/10.5281/zenodo.13828383). For detailed information about the experiment and dataset, see the [associated article](../README.md#citation).

### Download Instructions

Choose one of the following options to download and extract the sample data. After extraction, verify that your directory structure matches the [layout shown below](#expected-directory-structure).

#### Option 1: Direct Downloads

Download the files manually from the links below, then extract each ZIP file into the `data/` directory:

- [Sample Videos](https://zenodo.org/records/17924857/files/sample_videos.zip?download=1) — 26.8 GB (MD5: `cbe3b6f6c1dc5e9dce3406b4c1162b6f`)
- [Orthophotos](https://zenodo.org/records/17924857/files/orthophotos.zip?download=1) — 1.8 GB (MD5: `31a2712a456a22b3db8d5513e75da8b2`)
- [Segmentations](https://zenodo.org/records/17924857/files/segmentations.zip?download=1) — 24.9 KB (MD5: `c39304c88ddf39f3e0a40b8621d384b9`)
- [Master Frames](https://zenodo.org/records/17924857/files/master_frames.zip?download=1) — 248.6 MB (MD5: `4af9a065be7bcc4ac09ac45a9ce1af71`)

#### Option 2: Command-Line Download

Run the following commands from the `data/` directory to automatically download, extract, and clean up:

**Note for macOS users:** Replace `wget` with `curl -L` and `-O` with `-o`, or install wget via [Homebrew](https://brew.sh/): `brew install wget`

**Note for Windows users:** Install wget via [Chocolatey](https://chocolatey.org/) (`choco install wget`), or use [Option 1](#option-1-direct-downloads).

```bash
# Download and extract sample videos
wget "https://zenodo.org/records/17924857/files/sample_videos.zip?download=1" -O sample_videos.zip
unzip sample_videos.zip && rm sample_videos.zip

# Download and extract orthophotos
wget "https://zenodo.org/records/17924857/files/orthophotos.zip?download=1" -O orthophotos.zip
unzip orthophotos.zip && rm orthophotos.zip

# Download and extract segmentations
wget "https://zenodo.org/records/17924857/files/segmentations.zip?download=1" -O segmentations.zip
unzip segmentations.zip && rm segmentations.zip

# Download and extract master frames
wget "https://zenodo.org/records/17924857/files/master_frames.zip?download=1" -O master_frames.zip
unzip master_frames.zip && rm master_frames.zip
```

### Expected Directory Structure

After downloading and extracting the files, your `data/` directory should have the following structure:

```text
geo-trax/
└── data/
    ├── master_frames/
    │   ├── A.png
    │   ├── A.txt
    │   ├── ...
    │   ├── U.png
    │   └── U.txt
    ├── orthophotos/
    │   ├── A_center.txt
    │   ├── A.png
    │   ├── ...
    │   ├── ortho_parameters.txt
    │   ├── ...
    │   ├── U_center.txt
    │   └── U.png
    ├── README.md    
    ├── results-full/
    ├── results-pixel/    
    ├── sample_videos/
    │   ├── A_D1_2022-10-07_PM5_60s.mp4
    │   ├── ...
    │   └── U_D10_2022-10-07_PM5_60s.mp4
    ├── segmentations/
    │   ├── A.csv
    │   ├── A.png          ← generated (see note below)
    │   ├── ...
    │   ├── U.csv
    │   └── U.png          ← generated (see note below)
    ├── U_video_cut.csv
    └── U_video_cut.mp4
```

### Generate Segmentation Overlay PNGs

> **Note:** The segmentation visualization PNGs (`segmentations/*.png`) are **not** included in the downloaded `segmentations.zip`. Generate them locally after downloading the orthophotos and segmentation CSVs:
>
> ```bash
> python tools/viz_segmentations.py data/orthophotos/ -sf data/segmentations/
> ```
>
> This overlays each CSV's lane polygons and section labels onto the corresponding orthophoto and writes the annotated PNG to `data/segmentations/`.
