# Aggregate

`geotrax aggregate` merges the georeferenced CSVs of many drones and flights into one dataset, organized by date, location and flight session. It is the last step of a multi-drone campaign such as the Songdo experiment, where several drones covered the same intersection at the same time.

## Examples

```bash
# Write the dataset to a DATASET/ folder next to PROCESSED/
geotrax aggregate path/to/PROCESSED/

# Write it somewhere else
geotrax aggregate path/to/PROCESSED/ --output-folder path/to/output/

# Verbose console output and a log file in a chosen folder
geotrax aggregate path/to/PROCESSED/ --verbose --log-path path/to/logs/
```

## Expected layout

Grouping is read from the folder path of each result, which must follow `<date>/D<n>/<session>/<output folder>/`:

```text
PROCESSED/
└── 2022-10-07/          # date
    └── D1/              # drone: 'D' plus a number
        └── PM1/         # flight session
            └── results/
                ├── A1.csv
                └── A2.csv
```

The location ID is the leading letters of the clip name (`A1` → `A`). All clips of one location, date and session, from every drone, form one group. This is the layout of the [recommended project structure](https://github.com/rfonod/geo-trax#real-world-deployment-the-songdo-experiment). The results must carry timestamps, which come from a [flight log](own-data.md#flight-logs) per video.

## What it does

1. Finds every georeferenced CSV under the input folder and groups them by date, location ID and session.
2. Concatenates each group, offsetting each file's `Vehicle_ID` by the highest ID before it so IDs stay unique across drones, and adds a `Drone_ID` column.
3. Reduces each `Timestamp` to its time of day (`Local_Time`, `HH:MM:SS.fff`) and standardizes the column order and types (lane numbers become strings). No time zone is converted, so `Local_Time` is local only if the flight logs are; the date is kept in the folder and file names.
4. Writes one CSV per group, plus one ZIP archive per date and location for distribution.

## Outputs

```text
DATASET/
├── 2022-10-07_A/
│   ├── 2022-10-07_A_AM1.csv   # one CSV per session, all drones merged
│   └── 2022-10-07_A_PM1.csv
└── 2022-10-07_A.zip           # the CSVs of 2022-10-07_A/, for distribution
```

Each dataset CSV has the columns

```text
Vehicle_ID, Local_Time, Drone_ID, Ortho_X, Ortho_Y, Local_X, Local_Y, Latitude, Longitude, Vehicle_Length, Vehicle_Width, Vehicle_Class, Vehicle_Speed, Vehicle_Acceleration, Road_Section, Lane_Number, Visibility[, Is_Interpolated]
```

with the meanings of the [per-video CSV](georeference.md#trajectories-video_filecsv), except that `Local_Time` replaces `Timestamp` and `Frame_Number`, and `Drone_ID` is the number of the `D<n>` folder. Rows are sorted by `Vehicle_ID`, then `Local_Time`.

The dataset CSVs can be plotted with [`geotrax plot`](plot.md) and exported to GIS formats with [`geotrax export`](export.md). To trace a dataset `Vehicle_ID` back to its source video and original ID, use [`tools/find_source_id.py`](https://github.com/rfonod/geo-trax/blob/main/tools/find_source_id.py).

## Rules and skipped files

- Drone folders must be named `D` plus a number; results elsewhere are skipped with a warning.
- The output folder in [`output.folder`](../reference/config/output.md#output.folder) must be a relative name, since grouping relies on the per-video folder layout; an absolute output folder is rejected.
- Results without a `Timestamp` column (georeferenced without a flight log) are skipped, since their rows cannot be time-aligned with the other drones of the group.
- Empty (header-only) results are skipped, so they cannot disturb the `Vehicle_ID` offsets.
- Rows whose timestamp is undefined (frames missing from the flight log, written as `0000-00-00 00:00:00.000`) are dropped with a warning; the rest of the clip is kept.
- `Is_Interpolated` is kept when present; rows from runs without interpolation are marked `0`.
- Road sections and lane numbers left empty by georeferencing without a segmentation file stay empty.

All options are in the [CLI reference](../reference/cli/aggregate.md).
