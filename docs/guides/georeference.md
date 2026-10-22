# Georeference

`geotrax georeference` is stage 2. It registers the video's stabilization reference frame against a north-up orthophoto of the same location, maps every stabilized track point through that homography, and writes a per-vehicle CSV with orthophoto-pixel, local-CRS and WGS84 coordinates, real-world dimensions, speed, acceleration, and (with a segmentation file) road section and lane. It needs the tracks from [extract](extract.md).

## Examples

```bash
# Orthophotos from a given folder (master frames and segmentations from its sub-folders)
geotrax georeference path/to/video.mp4 -orf path/to/orthophotos

# A custom master-frames folder, recomputing the cached master -> orthophoto homography
geotrax georeference path/to/video.mp4 -mf path/to/master_frames --recompute

# A custom segmentation folder, and orthophoto parameters from a text file
geotrax georeference path/to/video.mp4 -osf path/to/segmentations --geo-source text-file

# GeoTIFF metadata as the parameter source, without master frames
geotrax georeference path/to/video.mp4 --geo-source metadata-tif --no-master

# A different feature detector for the registration
geotrax georeference path/to/video.mp4 -orf path/to/orthophotos --geo-detector sift
```

## Inputs

Each video is tied to its assets by its **location ID**, the leading letters of its filename (`A1.mp4` → `A`):

| Asset | Required | Default location | Flag |
|---|---|---|---|
| Orthophoto `A.png` plus its parameters (`A.txt`, or a georeferenced `A.tif`) | yes | `ORTHOPHOTOS/` next to the `PROCESSED/` folder that holds the video | `--ortho-folder`, `-orf` |
| Master frame `A.png` | unless `--no-master` | `<ortho-folder>/master_frames/` | `--master-folder`, `-mf` |
| Lane and road-section polygons `A.csv` | no | `<ortho-folder>/segmentations/` | `--segmentation-folder`, `-osf` |
| Flight log `A1.csv` next to the video | no | | |

`--geo-source` picks where the orthophoto's geographic parameters come from: `metadata-tif` (a georeferenced GeoTIFF), `text-file` (`A.txt`), or `center-text-file` (a shared `ortho_parameters.txt` plus a per-location `A_center.txt`); by default the source is detected from the files present. The orthophoto's CRS must match [`georef.transformation.source_crs`](../reference/config/georef.md#georef.transformation.source_crs) (WGS84 by default), and local coordinates are written in [`target_crs`](../reference/config/georef.md#georef.transformation.target_crs). With a flight log, every row also gets a timestamp. The sample data and how to obtain the full Songdo assets are described in [data/README.md](https://github.com/rfonod/geo-trax/blob/main/data/README.md).

## Master frames

When georeferencing, geo-trax can route each video's homography through a shared *master frame* per location ID. A master frame is a high-quality, near-nadir BEV frame chosen once per location (see [`tools/find_master_frames.py`](https://github.com/rfonod/geo-trax/blob/main/tools/find_master_frames.py)) and used instead of registering every video's reference frame directly against the orthophoto. The mapping is split into two homographies:

- `reference → master`, recomputed for every video;
- `master → orthophoto`, computed **once per location ID and cached** next to the master frame.

This gives two benefits:

- **Speed**: the expensive cross-domain `master → orthophoto` registration runs once and is reused across every drone and flight at that location, instead of once per video.
- **Consistency and robustness**: every video is matched against the *same* master frame. This same-modality BEV-to-BEV registration is far more reliable than a direct BEV-to-orthophoto match, so trajectories from different drones, altitudes and viewpoints resolve into one coherent coordinate system.

The cache is validated by hashes of the master image, the orthophoto and the [`georef.matching`](../reference/config/georef.md#georefmatching) settings, so it is recomputed when any of them changes. The exceptions are `gpu`, `gpu_device_id` and `device`, which only choose where the matching runs, so switching machines or devices reuses the cache. Caches written before v1.4.2 carry only the master-image hash; they are still accepted, with a warning that the other two could not be checked, so pass `--recompute` if they changed.

Master frames are enabled by default. Disable them with `--no-master`, or force the cached `master → orthophoto` homography to be recomputed with `--recompute`.

## Reference frame

The georeferencing anchor must be the same frame stabilization used. By default geo-trax reads it from the run metadata that `extract` wrote, so nothing needs setting. Passing `--ref-frame` overrides it, with a warning if it disagrees with the recorded anchor.

## Registration settings

The registration defaults to RootSIFT (`rsift`) rather than the ORB used for stabilization: matching a drone frame against an orthophoto crosses years, seasons and lighting, which is much harder than matching consecutive video frames. Matching parameters live in [`georef.matching`](../reference/config/georef.md#georefmatching) and kinematic smoothing in [`georef.filtering`](../reference/config/georef.md#georeffiltering). The detector can be changed with `--geo-detector`; read [Feature detectors](feature-detectors.md) first, since most learned detectors cannot handle the arbitrary rotation between a drone frame and a north-up orthophoto. The registration can be CUDA-accelerated with `--geo-gpu` (ORB only; see [GPU acceleration](gpu.md)).

## Outputs

Written to the same output folder as the tracks (default `results/`).

### Trajectories: `video_file.csv`

One row per vehicle per frame (`<stem><georeferenced_postfix>.csv`):

```text
Vehicle_ID, [Timestamp,] Frame_Number, Ortho_X, Ortho_Y, Local_X, Local_Y, Latitude, Longitude, Vehicle_Length, Vehicle_Width, Vehicle_Class, Vehicle_Speed, Vehicle_Acceleration, Road_Section, Lane_Number, Visibility[, Is_Interpolated]
```

- `Vehicle_ID`: unique vehicle identifier.
- `Timestamp`: frame time (`YYYY-MM-DD HH:MM:SS.ms`); present only when a flight-log CSV with timestamps sits next to the video.
- `Frame_Number`: video frame index of the row.
- `Ortho_X`, `Ortho_Y`: vehicle center in orthophoto pixels.
- `Local_X`, `Local_Y`: vehicle center in the local projected CRS (EPSG:5186 in the Songdo experiment).
- `Latitude`, `Longitude`: vehicle center in WGS84.
- `Vehicle_Length`, `Vehicle_Width`: estimated dimensions in meters.
- `Vehicle_Class`: 0 car (including vans), 1 bus, 2 truck, 3 motorcycle.
- `Vehicle_Speed`: speed in km/h.
- `Vehicle_Acceleration`: acceleration in m/s².
- `Road_Section`, `Lane_Number`: road segment and lane; empty without a segmentation file.
- `Visibility`: whether the bounding box is fully inside the frame.
- `Is_Interpolated` (optional): present when extraction ran with `--interpolate`; `1` marks a row interpolated to fill a frame gap.

Trajectories shorter than [`georef.filtering.min_traj_length`](../reference/config/georef.md#georef.filtering.min_traj_length) points are dropped. The CSV is written last, after the homography, so its presence marks a finished run.

### Georeferencing transform: `video_file_geo_transf.txt`

The 3 × 3 homography from the video's reference frame to the orthophoto, as nine comma-separated values:

```text
h11, h12, h13, h21, h22, h23, h31, h32, h33
```

!!! note "Reproducibility"
    Video decoding can differ between FFmpeg versions bundled with different OpenCV builds, which can shift frame extraction and so the georeferencing result. For identical outputs across machines, keep the OpenCV and FFmpeg versions the same.

All options are in the [CLI reference](../reference/cli/georeference.md).
