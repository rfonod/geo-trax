# Geo-trax

**Geo-trax** (GEO-referenced TRAjectory eXtraction) turns bird's-eye view drone video into georeferenced, lane-resolved vehicle trajectories. It detects and tracks vehicles with YOLO, removes drone motion by stabilizing every frame against a reference frame, and registers that reference frame to an orthophoto, so each vehicle ends up with WGS84 and local-CRS coordinates, speed, acceleration, estimated dimensions, and a lane.

## Demo

<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/gOGivL9FFLk" title="Geo-trax: georeferenced vehicle trajectories from drone imagery" loading="lazy" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Geo-trax applied to the Songdo experiment: 10 drones over 20 intersections, about 700,000 trajectories. To try the detector without installing anything, use the [interactive demo on Hugging Face Spaces](https://huggingface.co/spaces/rfonod/geo-trax).

## Install

Geo-trax needs Python 3.11 to 3.13.

```bash
python -m pip install geo-trax
```

| Extra | Adds | Install |
|---|---|---|
| `sahi` | SAHI sliced inference for small objects (`--sahi`) | `python -m pip install 'geo-trax[sahi]'` |
| `export` | ONNX model export | `python -m pip install 'geo-trax[export]'` |
| `dev` | Test and lint tooling | `python -m pip install 'geo-trax[dev]'` |

The default detector downloads from [Hugging Face](https://huggingface.co/rfonod/geo-trax) on first use, and geo-trax checks PyPI for a newer release at most once a day ([opt out](guides/configuration.md#update-check)). Other environments (uv, conda, Poetry) and editable installs are covered in the [README](https://github.com/rfonod/geo-trax#install); CUDA-accelerated image matching has its own [guide](guides/gpu.md).

## Quickstart

```bash
geotrax batch video.mp4 --no-geo                              # pixel-coordinate trajectories, no orthophoto needed
geotrax batch video.mp4 -orf path/to/orthophotos --no-master  # full pipeline: extract, georeference, visualize, plot
geotrax batch path/to/PROCESSED/                              # a whole project tree
geotrax aggregate path/to/PROCESSED/                          # merge drones and flights into one dataset
geotrax export path/to/DATASET/                               # GIS layers for QGIS; add --format geojson for kepler.gl
```

`--no-master` registers each video directly against the orthophoto; with per-location master frames, drop it and see the [georeference guide](guides/georeference.md#master-frames) for why they help. The flag-free project-tree commands rely on the [recommended folder layout](guides/own-data.md#folder-layout); any other layout works with explicit path flags.

!!! tip "Georeferencing your own footage"
    [Your own data](guides/own-data.md) lists the files each location needs (orthophoto, master frame, lane polygons, flight logs) and the defaults to change outside South Korea, above all `target_crs`, which the speeds and dimensions are measured in.

## Where to go next

- **[Guides](guides/batch.md)**: one page per pipeline stage; setting up [configuration](guides/configuration.md) and [your own data](guides/own-data.md); and topics: [detection and tracking](guides/detection-tracking.md), feature detectors, SAHI and GPU acceleration.
- **[Config reference](reference/config/index.md)**: every key of the pipeline config with its default, generated from the shipped file.
- **[CLI reference](reference/cli/index.md)**: every flag of every `geotrax` subcommand and the config key it overrides.
- **[API reference](reference/api/index.md)**: the registration, config and file helpers that the stages share.
- **[Citation](citation.md)**: the journal article, the software archive and the datasets.
