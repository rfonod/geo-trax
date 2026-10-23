# Batch processing

`geotrax batch` is the primary entry point. It runs every stage in order for one video or for every video below a folder:

1. **Extract**: detection, tracking and stabilization ([guide](extract.md)).
2. **Georeference**: registration against an orthophoto ([guide](georeference.md)), unless `--no-geo`.
3. **Visualize**: annotated videos ([guide](visualize.md)).
4. **Plot**: trajectory and distribution plots ([guide](plot.md)).

[Aggregate](aggregate.md) and [export](export.md) are not part of `batch`; run them afterwards on the finished tree.

## Examples

```bash
# Full pipeline on one video, with lane IDs in the annotated video
# (master frames and segmentations are read from the orthophoto folder's sub-folders)
geotrax batch path/to/video.mp4 -orf path/to/orthophotos --show-lanes

# The same without master frames
geotrax batch path/to/video.mp4 -orf path/to/orthophotos --no-master

# Pixel coordinates only, with class names and confidences in the annotated video
geotrax batch data/U_video_cut.mp4 --no-geo --show-class-names --show-conf

# Preview what a directory run would do, without executing anything
geotrax batch path/to/PROCESSED/ --dry-run --verbose

# Recompute everything in a tree, without confirmation prompts
geotrax batch path/to/PROCESSED/ --overwrite --yes

# Use a preset and skip videos whose name contains 'test' or 'tmp'
geotrax batch path/to/videos/ -c lenient --exclude-patterns test tmp --no-geo
```

## Re-runs and stage selection

Each stage first checks whether its output already exists and skips the video if it does, so an interrupted run resumes where it stopped. Tracks, transforms and georeferenced CSVs are written atomically (to a hidden temporary name, then renamed), so one that exists is always complete. Annotated videos are the exception: one stopped with `q` or Ctrl+C is kept as a partial video and counts as done, so render it again with `--viz-only --overwrite`. Pass `--overwrite` to recompute, and `--yes` to skip the confirmation.

To run or repeat a subset of the stages:

| Flag | Runs |
|---|---|
| `--no-geo` | everything except georeferencing |
| `--geo-only` | georeferencing only (needs the tracks from a previous run) |
| `--viz-only` | visualization only (needs the tracks from a previous run) |
| `--plot-only` | plotting only |

Georeferencing and visualization both need the tracks file from extraction; if it is missing, that stage is skipped for the video with an error, but the video is not counted as failed (see [Exit status](#exit-status)). A failure in one video does not stop the batch: the video is marked failed, its later stages are skipped, and the run continues.

## Directory mode

Given a folder, `batch` collects every video below it. Two filters narrow the scan:

- `--folders-exclude` skips folders by name, together with everything nested inside them. The configured output folder (default `results`) is always excluded on top of this list, so the annotated videos of an earlier run are never picked up as input.
- `--exclude-patterns` skips videos whose filename contains any of the given substrings.

In directory mode `--cut-frame-right` is ignored (with a warning if you passed it), so every video is processed to its last frame.

For multi-drone projects, the [recommended folder layout](own-data.md#folder-layout) lets `batch` find the orthophotos, master frames and segmentations without any path flag: it walks up from each video to the `PROCESSED` folder and looks for a sibling `ORTHOPHOTOS` folder.

## Exit status

| Code | Meaning |
|---|---|
| `0` | every video succeeded (or was skipped as already done) |
| `1` | at least one video failed; the failures are listed at the end |
| `130` | interrupted with Ctrl+C or `q` in the preview window; the failures so far are listed |

A stage skipped because its input is missing is logged as an error but is not a failure. This covers `--geo-only` or `--viz-only` on videos that were never extracted, and videos in which nothing was detected, since those leave no tracks file. After a run that ends with `0`, scripts should still check the log for errors.

`--dry-run` does not create a log file, so it is safe to run freely before a long job. All options are listed in the [CLI reference](../reference/cli/batch.md).
