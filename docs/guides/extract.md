# Extract

`geotrax extract` is stage 1. It detects vehicles in every frame with YOLO, links the detections into tracks with the selected multi-object tracker, and stabilizes the tracks by registering each frame to a fixed reference frame with [Stabilo](https://github.com/rfonod/stabilo). It also estimates each vehicle's length and width in pixels. The output is in image coordinates; [georeferencing](georeference.md) turns it into real-world coordinates.

## Examples

```bash
# Default settings
geotrax extract path/to/video.mp4

# Only the first two vehicle classes (car, bus), with a locally copied config
geotrax extract path/to/video.mp4 -c default_copy.yaml --classes 0 1

# Skip the first 100 frames and stop after frame 500
geotrax extract path/to/video.mp4 --cut-frame-left 100 --cut-frame-right 500

# Rename the class labels (overrides the model's own names)
geotrax extract path/to/video.mp4 -cn 0=vehicle 1=bus 2=truck 3=bike

# Fill per-track detection gaps with linear interpolation
geotrax extract path/to/video.mp4 --interpolate

# Sliced inference for small objects (needs the sahi extra)
geotrax extract path/to/video.mp4 --sahi
```

## What controls what

| Concern | Where |
|---|---|
| Detection model | `--model`, or [`extraction.model`](../reference/config/extraction.md#extraction.model): a local `.pt` path or an `hf://<org>/<repo>/<file>.pt` reference ([details](detection-tracking.md#detection-model)) |
| Classes | `--classes` to filter, `--class-names` to rename ([details](detection-tracking.md#classes)) |
| Detection thresholds | `--conf`, or [`ultralytics`](../reference/config/ultralytics.md); per class with [`extraction.class_conf`](../reference/config/extraction.md#extraction.class_conf) ([details](detection-tracking.md#confidence-thresholds)) |
| Tracker | [`tracker.active`](../reference/config/tracker.md#tracker.active), and that tracker's block in `tracker` ([comparison](detection-tracking.md#trackers)) |
| Stabilization | [`stabilo`](../reference/config/stabilo.md); the detector with `--stab-detector` (see [Feature detectors](feature-detectors.md)) |
| Short tracks | [`extraction.min_track_length`](../reference/config/extraction.md#extraction.min_track_length) (no flag; use `--set min_track_length=5`) |
| Dimension estimation | [`extraction.dimension_estimation`](../reference/config/extraction.md#extractiondimension_estimation), including the ground sample distance `gsd` |

The `gsd` default (0.02725 m/px) is tuned for a DJI Mavic 3 at 140 to 150 m and 4K; re-tune it for other altitudes or cameras, as described in [Your own data](own-data.md#settings-to-adapt).

The stabilization reference frame is the first processed frame, `cut_frame_left`. It is recorded in the run metadata, and [georeferencing](georeference.md) and [visualization](visualize.md) read it from there, so they stay consistent with the stabilized coordinates even if they run with a different frame range.

## Outputs

For an input `video_file.mp4`, the outputs go to `results/` next to the video by default. The folder (`--output-folder`) and every filename postfix are set in the [`output`](../reference/config/output.md) section.

### Tracks: `video_file.txt`

One row per vehicle per frame (`<stem><tracks_postfix>.txt`):

```text
frame_id, vehicle_id, x_c(unstab), y_c(unstab), w(unstab), h(unstab), x_c(stab), y_c(stab), w(stab), h(stab), class_id, confidence, vehicle_length, vehicle_width
```

- `frame_id`: frame number (0, 1, ...).
- `vehicle_id`: unique vehicle identifier (1, 2, ...).
- `x_c(unstab)`, `y_c(unstab)`, `w(unstab)`, `h(unstab)`: bounding-box center and size in the original frame.
- `x_c(stab)`, `y_c(stab)`, `w(stab)`, `h(stab)`: the same in the stabilized (reference) frame.
- `class_id`: 0 car (including vans), 1 bus, 2 truck, 3 motorcycle.
- `confidence`: detection confidence (0 to 1).
- `vehicle_length`, `vehicle_width`: estimated vehicle dimensions in pixels, one value per vehicle; `nan` when no estimate could be made (short, stationary or edge-bound tracks).
- `is_interpolated` (optional 15th column): present only with `--interpolate` (`extraction.interpolate: true`); `0` for a real detection, `1` for a row interpolated to fill a frame gap. Gaps longer than the active tracker's `track_buffer` are left unfilled, since the tracker would not have kept a lost track's ID across a longer occlusion.

Each track has a single class, chosen by a confidence-weighted vote over its detections, so a track can hold rows whose own detection was another class.

### Stabilization transforms: `video_file_vid_transf.txt`

One 3 × 3 homography per frame (`<stem><stab_transform_postfix>.txt`):

```text
frame_id, h11, h12, h13, h21, h22, h23, h31, h32, h33
```

Each matrix maps frame `frame_id` to the reference frame. The first row is frame `cut_frame_left + 1`, since the reference frame itself has no transform.

### Run metadata: `video_file.yaml`

A record of the run (geo-trax version, time, source video, config file, model and class names) and of every setting as the run used it, with all CLI flags and `--set` overrides applied, so it records the run rather than the file it started from. See [What a run records](configuration.md#what-a-run-records).

A tracking or saving failure leaves no tracks file and makes `extract` exit with status 1, so `batch` never treats a failed run as done. A video in which nothing was detected also leaves no tracks file, but `extract` exits with status 0 and the run metadata records zero vehicles; `batch`, `georeference` and `visualize` then treat the video as done, with nothing to process. All options are in the [CLI reference](../reference/cli/extract.md).
