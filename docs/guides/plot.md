# Plot

`geotrax plot` draws trajectory maps and distribution charts from the results of [extract](extract.md) and [georeference](georeference.md). It accepts a video (its results are looked up in the output folder), a `.txt` tracks file, a georeferenced or aggregated `.csv`, or a folder searched for any of these.

## Examples

```bash
# One video, pixel-coordinate results only
geotrax plot path/to/video.mp4

# Georeferenced trajectories, drawn over the orthophoto
geotrax plot path/to/results/video.csv -orf path/to/orthophotos

# Every result in a folder, merged per location
geotrax plot path/to/videos/ --plot-aggregate

# Over the lane segmentation overlays as well, hiding buses
geotrax plot path/to/videos/ -orf path/to/orthophotos -osf path/to/segmentations --plot-segmentations --plot-class-filter 1

# Rename the class labels
geotrax plot path/to/video.mp4 -cn 0=vehicle 1=bus 2=truck 3=bike
```

The plotting flags carry a `--plot-` prefix so that `geotrax batch` accepts them unchanged next to the visualization flags; there, `--plot-only` regenerates the plots from existing results.

## What it draws

| Plot | Needs |
|---|---|
| Trajectories in image coordinates (original and stabilized) | tracks |
| Trajectories in the georeferenced coordinate systems, plus over the orthophoto image | georeferenced results (and the orthophoto) |
| The same over the lane segmentation overlay | `--plot-segmentations` and the segmentation PNG |
| Speed and acceleration distributions, separately and jointly | georeferenced results |
| Class distribution | georeferenced results |
| Vehicle length and width distributions | georeferenced results (meters), else tracks (pixels) |
| Speed and acceleration of one vehicle over time, instead of the distributions | `--id <vehicle ID>` and georeferenced results |

Vehicles slower than [`plotting.stationary_speed_cutoff`](../reference/config/plotting.md#plotting.stationary_speed_cutoff) km/h are left out of the speed and acceleration plots. `--plot-points` draws discrete points instead of lines, and `--plot-class-filter` hides classes by ID.

## Aggregation

With a folder and `--plot-aggregate`, all videos that share a location ID are drawn into one plot per location, each source in its own color from [`plotting.colors`](../reference/config/plotting.md#plotting.colors). Aggregated dataset CSVs from [`geotrax aggregate`](aggregate.md) are plotted per file, since they are already merged.

## Output

Plots are saved as PDF files in `plots/` inside the output folder (default `results/plots/`); `--plot-show` opens them in a window instead of, or as well as, saving them. Result files whose name contains any string in [`plotting.skip_filenames_with`](../reference/config/plotting.md#plotting.skip_filenames_with) are ignored.

All options are in the [CLI reference](../reference/cli/plot.md).
