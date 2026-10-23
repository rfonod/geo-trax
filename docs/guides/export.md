# Export

`geotrax export` converts georeferenced trajectories into vector files that open directly in QGIS, ArcGIS or any GDAL-based tool, and as GeoJSON in web tools such as kepler.gl. It reads the per-video CSVs from [georeference](georeference.md) and the dataset CSVs from [aggregate](aggregate.md). Each vehicle becomes one LineString (its trajectory), or each trajectory row becomes one Point. It is a standalone command; `geotrax batch` does not run it.

## Examples

```bash
# One video's trajectories as a GeoPackage of LineStrings, next to the CSV
geotrax export path/to/results/video.csv

# Via the video path (its CSV is looked up in the output folder), as GeoJSON points
geotrax export path/to/video.mp4 --format geojson --geometry points

# A whole aggregated dataset in the local projected CRS, into one folder
geotrax export path/to/DATASET/ --crs local -of path/to/GIS/
```

The input can be a CSV, a video, or a folder. A folder is searched recursively, and a CSV qualifies when its header has `Vehicle_ID` plus the coordinate columns of the chosen CRS, so flight logs and other CSVs in the tree are skipped. With `--output-folder`, the folder layout below the input is mirrored.

## Options

| Option | Values | Default |
|---|---|---|
| `--format` | `gpkg` (GeoPackage), `geojson` | `gpkg` |
| `--geometry` | `lines`: one LineString per vehicle with summary attributes; `points`: one Point per row with every CSV column | `lines` |
| `--crs` | `wgs84`: `Latitude`/`Longitude` in EPSG:4326; `local`: `Local_X`/`Local_Y` in [`georef.transformation.target_crs`](../reference/config/georef.md#georef.transformation.target_crs) | `wgs84` |

## Output files

`<stem>_<lines|points>[_local].<gpkg|geojson>`, for example `video_lines.gpkg` or `video_points_local.gpkg`, each holding one layer of the same name. The `_local` suffix marks `--crs local`, so the two CRSs never overwrite each other.

`lines` holds one LineString per vehicle, in time order, with the attributes:

- `Vehicle_ID`, `Vehicle_Class`, and `Drone_ID` for aggregated datasets;
- `Num_Points`, the number of points on the line;
- `Start_Time`/`End_Time`, from `Timestamp` for per-video CSVs (empty when the video had no flight log) and from `Local_Time` for aggregated datasets;
- `Start_Frame`/`End_Frame` for per-video CSVs;
- `Mean_Speed`/`Max_Speed` in km/h;
- the median `Vehicle_Length`/`Vehicle_Width` in meters.

Each line's vertices are ordered by `Frame_Number` for per-video CSVs and by `Local_Time` for aggregated datasets. A vehicle with fewer than two valid points cannot form a line and is skipped, with a count in the log; rows without valid coordinates are dropped with a warning.

GeoJSON ([RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946)) expects WGS84, so `--crs local` with `--format geojson` is written but warned about. Prefer GeoPackage for large datasets: GeoJSON is text and grows quickly in points mode.

## Viewing exported files

| Format | What it is | Best for |
|---|---|---|
| **GeoPackage** (`.gpkg`, default) | [OGC open standard](https://www.geopackage.org/): a single SQLite file that can hold large datasets in any coordinate system | Desktop GIS, large datasets, the local projected CRS (`--crs local`) |
| **GeoJSON** (`.geojson`) | [RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) plain-text JSON in WGS84 | Web maps, quick sharing, small to medium datasets |

Both open in QGIS, ArcGIS Pro, GDAL/OGR, Python (GeoPandas), R (`sf`), and DuckDB (spatial extension); GeoJSON also opens in most web map tools. The examples below use the [sample export](https://github.com/rfonod/geo-trax/blob/main/data/README.md#gis-export) shipped in `data/results-full/` (144 vehicle trajectories from the 5-second demo clip), where each LineString is one vehicle with the [attributes listed above](#output-files).

### In the browser (no install)

- **GitHub**: open [`U_video_cut_lines.geojson`](https://github.com/rfonod/geo-trax/blob/main/data/results-full/U_video_cut_lines.geojson) in the repository. GitHub renders any `.geojson` file as an interactive map; click a line to see its attributes.
- **[geojson.io](https://geojson.io/)** (open source): drag and drop a `.geojson` file onto the map to view it on a basemap, inspect each feature's attributes, or edit it.
- **[kepler.gl](https://kepler.gl/demo)** (open source, OpenJS Foundation): made for large movement datasets. Data is processed locally in your browser.
    1. Open [kepler.gl/demo](https://kepler.gl/demo) and drag `U_video_cut_lines.geojson` into the **Add Data** dialog. A line layer appears on the basemap.
    2. In the layer panel, open the layer and set **Stroke Color** to be based on `Mean_Speed` (speed heat map) or `Vehicle_Class`.
    3. For playback, also add the georeferenced CSV (`data/results-full/U_video_cut.csv`). kepler.gl builds a point layer from its `Latitude`/`Longitude` columns; under **Filters**, add a filter on `Timestamp` to get a time slider with a play button that animates the vehicles.

### On the desktop: QGIS

[QGIS](https://qgis.org/) is free and open source, for Windows, macOS and Linux.

1. Drag `U_video_cut_lines.gpkg` (or the `.geojson`) into the QGIS window.
2. Add a basemap: in the **Browser** panel, expand **XYZ Tiles** and double-click **OpenStreetMap**, then drag it below the trajectory layer in the **Layers** panel.
3. Color by attribute: right-click the layer, choose **Properties > Symbology**, select **Graduated** with value `Mean_Speed` (or **Categorized** with `Vehicle_Class`), and click **Classify**.
4. Click any trajectory with the **Identify Features** tool to see its attributes. A `_local` export (EPSG:5186 here) is reprojected on the fly, so it overlays the same basemap.

### In Python

GeoPandas, Matplotlib and Folium are already installed with geo-trax.

```python
import geopandas as gpd
import matplotlib.pyplot as plt

lines = gpd.read_file("data/results-full/U_video_cut_lines.gpkg")  # the .geojson reads the same way
print(lines.head())
print(lines.groupby("Vehicle_Class")["Mean_Speed"].mean())  # mean speed per class [km/h]

lines.plot(column="Mean_Speed", cmap="viridis", legend=True, figsize=(8, 8))  # static map colored by speed
plt.show()

# Interactive web map, saved as HTML to open in any browser (in a Jupyter notebook, a bare `m` shows it inline)
m = lines.explore(column="Vehicle_Class", categorical=True, tooltip=["Vehicle_ID", "Mean_Speed", "Vehicle_Length"])
m.save("U_video_cut_map.html")
```

### Converting to other formats

[GDAL's `ogr2ogr`](https://gdal.org/programs/ogr2ogr.html) (bundled with QGIS, or `conda install -c conda-forge gdal`) converts to anything GDAL writes, such as KML for Google Earth or a Shapefile for older GIS tools:

```bash
ogr2ogr -f KML U_video_cut_lines.kml data/results-full/U_video_cut_lines.gpkg
ogr2ogr -f "ESRI Shapefile" U_video_cut_lines.shp data/results-full/U_video_cut_lines.gpkg
```

All options are in the [CLI reference](../reference/cli/export.md).
