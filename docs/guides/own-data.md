# Your own data

Pixel-coordinate trajectories need nothing but the video: `geotrax batch video.mp4 --no-geo`. Georeferencing needs a few files per **location**, plus settings that suit your region. The [sample data](https://github.com/rfonod/geo-trax/blob/main/data/README.md) has every file described here for one Songdo intersection, so it is a useful template.

| What | Needed for | |
|---|---|---|
| Video names that start with a location ID | linking each video to the files below | [Location IDs](#location-ids) |
| An orthophoto per location, with its georeferencing | georeferencing | [Orthophotos](#orthophotos) |
| A master frame per location, or `--no-master` | georeferencing | [Master frames](#master-frames) |
| Lane polygons per location | `Road_Section`, `Lane_Number`, `--show-lanes` | [Lane segmentation](#lane-segmentation) |
| A flight log per video | `Timestamp`, and `geotrax aggregate` | [Flight logs](#flight-logs) |
| A local CRS for your region | correct local coordinates, speeds, accelerations and dimensions | [Settings to adapt](#settings-to-adapt) |

## Location IDs

geo-trax finds a video's orthophoto, master frame and segmentation by its **location ID**: the first run of letters in the filename, ended by a digit, `_` or `-`. `A1.mp4`, `A_D1_2022-10-07.mp4` and `A-north.mp4` all belong to location `A`, and `Gate3.mp4` to `Gate`. Keep the case consistent with the asset files: `a1.mp4` looks for `a.png`.

## Folder layout

Pass the folders explicitly with `-orf`, `-mf` and `-osf`, set them once in the [`input`](../reference/config/input.md) config section, or use the layout that geo-trax detects on its own. It walks up from each video to the nearest folder named `PROCESSED` (or `DATASET`) and uses its sibling `ORTHOPHOTOS`:

```text
project/
├── PROCESSED/                 # videos, in any sub-folders
│   └── .../A1.mp4, A1.csv     # a video and its flight log
└── ORTHOPHOTOS/
    ├── A.png                  # orthophoto of location A
    ├── A.txt                  # its georeferencing (or A.tif)
    ├── master_frames/
    │   └── A.png              # master frame (geo-trax adds its cache, A.txt)
    └── segmentations/
        ├── A.csv              # lane polygons
        └── A.png              # lane overlay, used by plots only
```

[`geotrax aggregate`](aggregate.md#expected-layout) also needs `<date>/D<n>/<session>/` folders below `PROCESSED`. The README describes the complete [project layout](https://github.com/rfonod/geo-trax#real-world-deployment-the-songdo-experiment) used in the Songdo experiment, from raw footage to dataset.

## Orthophotos

The orthophoto is a north-up aerial image of the location, `A.png`, that covers everything the drone sees, since registration needs the two to overlap. geo-trax also needs the coordinates of its pixels, from one of three sources:

| Source | Files | `--geo-source` |
|---|---|---|
| GeoTIFF | `A.tif` | `metadata-tif` |
| Parameter file | `A.png` and `A.txt` | `text-file` |
| Cut-out of a larger orthophoto | `A.png`, `A_center.txt` and a shared `ortho_parameters.txt` | `center-text-file` |

The source is picked from the files present. If files for two sources exist, the run stops and asks you to choose one with `--geo-source`.

!!! warning "Use WGS84 coordinates"
    `Latitude` and `Longitude` are computed directly from these parameters, so they must be in WGS84 degrees (EPSG:4326, the default [`georef.transformation.source_crs`](../reference/config/georef.md#georef.transformation.source_crs)). With a projected orthophoto, such as a GeoTIFF in a UTM zone, those columns would hold eastings and northings instead. Reproject such a file first, e.g. with GDAL: `gdalwarp -t_srs EPSG:4326 -r bilinear ortho_utm.tif A.tif`.

### GeoTIFF

geo-trax reads the georeferencing tags of `A.tif`. Image matching always uses `A.png`, so if it is missing, geo-trax converts the GeoTIFF and writes `A.png` next to it, which needs write access to the folder.

### Parameter file

`A.txt` holds four or six numbers, one per line; `#` starts a comment:

```text
126.6312291   # lng0: longitude of pixel (0, 0), the top-left corner
37.4010582    # lat0: latitude of pixel (0, 0)
1.814720e-07  # dlng: change in longitude per pixel to the right
-1.448040e-07 # dlat: change in latitude per pixel down (negative for a north-up image)
0             # skew_x (optional): change in longitude per pixel down
0             # skew_y (optional): change in latitude per pixel to the right
```

so that, for a pixel at column `x` and row `y` of `A.png`:

```text
longitude = lng0 + dlng * x + skew_x * y
latitude  = lat0 + dlat * y + skew_y * x
```

These are the six coefficients of a GDAL geotransform, in a different order: `gdalinfo` on a georeferenced image reports `lng0, lat0` as its *Origin* and `dlng, dlat` as its *Pixel Size*.

### Cut-outs of a larger orthophoto

When many locations are cut from one large orthophoto, as [`tools/subset_orthophoto.py`](https://github.com/rfonod/geo-trax/blob/main/tools/subset_orthophoto.py) does:

- `ortho_parameters.txt` holds the parameters of the large orthophoto, in the same format as `A.txt`, shared by every location;
- `A_center.txt` holds two numbers, the column and row of the cut-out's center in the large orthophoto;
- `A.png` is the square cut-out, possibly downscaled;
- [`georef.transformation.cutout_width_px`](../reference/config/georef.md#georef.transformation.cutout_width_px) is the side of the cut-out in pixels of the large orthophoto, before any downscaling. geo-trax derives the pixel size of `A.png` from it. The shipped value, 15000, is the Songdo one; if your cut-outs are not downscaled, set it to the width of `A.png`.

## Master frames

A master frame is one fixed frame per location that every video at that location is registered through; the [georeference guide](georeference.md#master-frames) explains why this helps. Master frames are on by default, and a missing `master_frames/A.png` stops georeferencing with an error. Either:

- go without: pass `--no-master`, or set [`georef.processing.use_master`](../reference/config/georef.md#georef.processing.use_master) to `false`, and each video is registered directly against the orthophoto; or
- provide one: any sharp frame typical of your flights at that location works, for example the first frame of a good clip:

    ```bash
    mkdir -p ORTHOPHOTOS/master_frames
    python -c "import cv2; ok, frame = cv2.VideoCapture('A1.mp4').read(); cv2.imwrite('ORTHOPHOTOS/master_frames/A.png', frame)"
    ```

    [`tools/find_master_frames.py`](https://github.com/rfonod/geo-trax/blob/main/tools/find_master_frames.py) picks the best frame per location from the flight logs instead. The `tools/` scripts are in the source repository, not in the PyPI package, so clone the repository to use them.

On first use geo-trax registers the master frame against the orthophoto and caches the result as `master_frames/A.txt`, so that folder must be writable.

## Lane segmentation

Optional. `segmentations/A.csv` holds one lane polygon per row, under this header:

```text
Section,Lane,tlx,tly,blx,bly,brx,bry,trx,try
1_1,1,7401,1342,5938,3247,6021,3313,7480,1403
1_1,2,7480,1403,6021,3313,6101,3378,7561,1471
```

- `Section`: the road-section label written to `Road_Section`, any text.
- `Lane`: the lane label written to `Lane_Number`.
- `tlx` to `try`: the top-left, bottom-left, bottom-right and top-right corners of the lane, in pixels of `A.png`.

Each lane is one quadrilateral, so split a curved lane into several rows with the same `Section` and `Lane`. Points outside every polygon get no section or lane; avoid overlapping polygons, since a point inside two of them gets only one. Columns after the tenth are ignored. `--plot-segmentations` also needs an overlay image `segmentations/A.png`, which [`tools/viz_segmentations.py`](https://github.com/rfonod/geo-trax/blob/main/tools/viz_segmentations.py) draws from the CSV.

## Flight logs

Optional, but [`geotrax aggregate`](aggregate.md) skips results without timestamps. The flight log is a CSV next to the video with the same name (`A1.csv` or `A1.CSV`), with at least these columns:

| Column | Content |
|---|---|
| `frame` | Video frame index, from 0. A log that starts at another number is shifted to 0, with a warning. |
| `timestamp` | Time of that frame, e.g. `2022-10-07 17:52:13.500`, in any format pandas can parse. |

Other columns are ignored by georeferencing. Timestamps are copied as they are into the `Timestamp` column; frames missing from the log get the placeholder `0000-00-00 00:00:00.000`, and `aggregate` drops those rows. `aggregate` keeps only the time of day (`Local_Time`) and converts no time zones, so write the log in local time. For DJI drones, the `tools/` scripts that merge the recorded segments and cut them into per-location clips also turn the drone's `.SRT` logs into these CSVs; the README walks through them under [From raw footage to trajectories](https://github.com/rfonod/geo-trax#real-world-deployment-the-songdo-experiment).

## Settings to adapt

A few defaults are tuned for the Songdo experiment:

| Key | Shipped value | Set it to |
|---|---|---|
| [`georef.transformation.target_crs`](../reference/config/georef.md#georef.transformation.target_crs) | `epsg:5186` (Korea 2000 / Central Belt 2010) | a projected CRS in meters for your area, e.g. its UTM zone: `epsg:326NN` north of the equator, `epsg:327NN` south |
| [`extraction.dimension_estimation.gsd`](../reference/config/extraction.md#extraction.dimension_estimation.gsd) | `0.02725` m/px (DJI Mavic 3, 140 to 150 m, 4K) | the ground sample distance of your footage |
| [`georef.transformation.cutout_width_px`](../reference/config/georef.md#georef.transformation.cutout_width_px) | `15000` | only with cut-outs; [see above](#cut-outs-of-a-larger-orthophoto) |

!!! warning "Set `target_crs` outside South Korea"
    `Local_X` and `Local_Y` are written in `target_crs`, and speeds, accelerations and vehicle dimensions are measured in it. EPSG:5186 is only accurate in South Korea; elsewhere every distance comes out too long, and geo-trax does not warn about it. A true 10 m measures 12.8 m in Vienna and 15.2 m in Los Angeles.

The GSD is used only to pick the frames for estimating vehicle dimensions (it converts [`r0`](../reference/config/extraction.md#extraction.dimension_estimation.r0) from meters to pixels); the dimensions themselves are measured through the orthophoto. To find it, divide a known ground length, such as a lane width, by its length in pixels in a video frame.

Set these in a [copied config](configuration.md#your-own-config), or for one run:

```bash
geotrax batch PROCESSED/ --set target_crs=epsg:32633 --set gsd=0.03
```
