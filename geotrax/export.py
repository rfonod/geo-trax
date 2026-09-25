#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
export.py - Export georeferenced trajectories to GIS formats (GeoPackage, GeoJSON).

Converts the georeferenced trajectory CSVs written by 'geotrax georeference', and the aggregated
dataset CSVs written by 'geotrax aggregate', into vector files that open directly in QGIS, ArcGIS,
kepler.gl, or any GDAL-based tool. Each vehicle becomes one LineString (its trajectory) or each
trajectory row becomes one Point.

Usage:
  geotrax export <input> [options]

Arguments:
  input : A georeferenced .csv file, a video file (its georeferenced .csv is looked up in the
          output folder, see cfg -> output), or a folder that is searched recursively for
          georeferenced and aggregated .csv files.

Options:
  -h, --help                  : Show this help message and exit.
  -f, --format <str>          : Output format: gpkg (GeoPackage) or geojson (default: gpkg).
  -g, --geometry <str>        : lines = one LineString per vehicle with per-vehicle summary attributes;
                                points = one Point per trajectory row with every CSV column (default: lines).
  -crs, --crs <str>           : wgs84 = Latitude/Longitude in EPSG:4326; local = Local_X/Local_Y in the
                                projected CRS of cfg -> georef -> transformation -> target_crs (default: wgs84).
  -of, --output-folder <path> : Folder for the exported files. Defaults to the folder of each input .csv;
                                for a folder input, the sub-folder layout below it is mirrored.
  -c, --cfg <path>            : Pipeline config, used for the output-folder layout (video inputs) and the
                                local CRS. Defaults to the bundled config.
  -st, --set <KEY=VALUE>      : Override a pipeline config value for this run; repeat for more than one.
  -lp, --log-path <str>       : Where to write logs: a directory or a full file path; defaults to a
                                platform-specific log directory.
  -v, --verbose               : Set print verbosity level to INFO (default: WARNING).

Examples:
  1. Export one video's trajectories as a GeoPackage of LineStrings (next to the CSV):
     geotrax export path/to/results/video.csv

  2. Export via the video path, as GeoJSON points:
     geotrax export path/to/video.mp4 --format geojson --geometry points

  3. Export an aggregated dataset folder in the local projected CRS into one folder:
     geotrax export path/to/DATASET/ --crs local -of path/to/GIS/

Output:
  '<stem>_<geometry>[_local].<gpkg|geojson>', e.g. 'video_lines.gpkg' or 'video_points_local.gpkg', holding
  one layer of the same name. The '_local' suffix marks '--crs local', so the two CRSs never overwrite each other.

Notes:
  - A .csv qualifies when its header has Vehicle_ID plus the coordinate columns of the chosen CRS, so
    drone flight logs and other CSVs in a scanned folder are skipped.
  - Lines are ordered by Frame_Number (per-video CSVs) or Local_Time (aggregated datasets). Vehicles
    with fewer than two valid points cannot form a line and are skipped with a count.
  - Line attributes: Vehicle_ID, Vehicle_Class, Drone_ID (aggregated datasets), Num_Points,
    Start_Time/End_Time, Start_Frame/End_Frame (per-video CSVs), Mean_Speed/Max_Speed [km/h], and the
    median Vehicle_Length/Vehicle_Width [m].
  - Rows without valid coordinates are dropped with a warning.
  - GeoJSON (RFC 7946) expects WGS84; '--crs local' with '--format geojson' is written but warned about.
    Use GeoPackage for large datasets: GeoJSON is text and grows quickly in points mode.
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from geotrax.utils.cli_utils import add_common_args, finalize_cli_args
from geotrax.utils.config_utils import load_config
from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.file_utils import atomic_output, build_result_path
from geotrax.utils.logging_utils import setup_logger

FORMATS = {'gpkg': ('GPKG', '.gpkg'), 'geojson': ('GeoJSON', '.geojson')}
COORDINATE_COLUMNS = {'wgs84': ('Longitude', 'Latitude'), 'local': ('Local_X', 'Local_Y')}
WGS84 = 'EPSG:4326'
UNDEFINED_TIMESTAMP = '0000-00-00 00:00:00.000'


def is_trajectory_csv(path: Path, crs_mode: str) -> bool:
    """Return True when the header of *path* has Vehicle_ID and the coordinate columns of *crs_mode*."""
    try:
        header = pd.read_csv(path, nrows=0).columns
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return {'Vehicle_ID', *COORDINATE_COLUMNS[crs_mode]}.issubset(header)


def find_input_csvs(source: Path, crs_mode: str, out_cfg: dict, logger: logging.Logger) -> list[Path]:
    """Resolve *source* (a .csv, a video, or a folder) to the trajectory CSVs to export.

    A video resolves to its georeferenced CSV in the output folder, never to the drone flight log
    that shares its stem next to it. A folder is scanned recursively, keeping only files that pass
    :func:`is_trajectory_csv`.
    """
    if source.is_dir():
        csv_files = sorted(
            p for p in source.rglob('*') if p.suffix.lower() == '.csv' and is_trajectory_csv(p, crs_mode)
        )
        if not csv_files:
            logger.critical(f"No georeferenced trajectory CSVs found under '{source}'.")
            sys.exit(1)
        return csv_files

    if source.suffix.lower() in VIDEO_FORMATS:
        csv_file = build_result_path(source, 'georeferenced', out_cfg)
        if not csv_file.exists():
            logger.critical(f"No georeferenced CSV for '{source}' at '{csv_file}'. Run 'geotrax georeference' first.")
            sys.exit(1)
    elif source.suffix.lower() == '.csv':
        csv_file = source
        if not csv_file.exists():
            logger.critical(f"Input file '{csv_file}' does not exist.")
            sys.exit(1)
    else:
        logger.critical(f"Unsupported input '{source}': expected a .csv file, a video file, or a folder.")
        sys.exit(1)

    if not is_trajectory_csv(csv_file, crs_mode):
        x_col, y_col = COORDINATE_COLUMNS[crs_mode]
        logger.critical(f"'{csv_file}' is not a georeferenced trajectory CSV (needs Vehicle_ID, {x_col}, {y_col}).")
        sys.exit(1)
    return [csv_file]


def drop_invalid_coordinates(df: pd.DataFrame, crs_mode: str, source: Path, logger: logging.Logger) -> pd.DataFrame:
    """Drop the rows whose coordinates for *crs_mode* are missing or not finite, warning with a count."""
    x_col, y_col = COORDINATE_COLUMNS[crs_mode]
    coords = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(coords.to_numpy(dtype=float)).all(axis=1)
    if not valid.all():
        logger.warning(f"Dropped {int((~valid).sum())} row(s) without valid {x_col}/{y_col} from '{source}'.")
    df = df.loc[valid].copy()
    df[[x_col, y_col]] = coords.loc[valid]
    return df


def time_order_column(df: pd.DataFrame) -> str | None:
    """Return the column that orders a vehicle's points: Frame_Number, else Local_Time, else Timestamp."""
    return next((c for c in ('Frame_Number', 'Local_Time', 'Timestamp') if c in df.columns), None)


def build_points(df: pd.DataFrame, crs_mode: str, crs: str) -> gpd.GeoDataFrame:
    """Build one Point per trajectory row, keeping every CSV column as an attribute."""
    x_col, y_col = COORDINATE_COLUMNS[crs_mode]
    geometry = gpd.points_from_xy(df[x_col], df[y_col])
    return gpd.GeoDataFrame(df.reset_index(drop=True), geometry=geometry, crs=crs)


def build_lines(df: pd.DataFrame, crs_mode: str, crs: str, source: Path, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Build one LineString per vehicle, with per-vehicle summary attributes.

    Points are ordered by :func:`time_order_column` within each vehicle, and the vertices are
    assembled in one vectorized ``shapely.linestrings`` call rather than per vehicle. The
    '0000-00-00 00:00:00.000' timestamp that georeferencing writes for frames missing from the
    flight log is treated as unknown for Start_Time/End_Time. Speed statistics ignore the empty
    speeds of the first points of a track.
    """
    x_col, y_col = COORDINATE_COLUMNS[crs_mode]
    order_col = time_order_column(df)
    df = df.sort_values(['Vehicle_ID', order_col] if order_col else ['Vehicle_ID'], kind='stable')

    counts = df.groupby('Vehicle_ID', sort=False).size()
    short = counts.index[counts < 2]
    if len(short):
        logger.info(f"Skipped {len(short)} vehicle(s) with fewer than 2 valid points in '{source}'.")
        df = df[~df['Vehicle_ID'].isin(short)]
    if df.empty:
        return gpd.GeoDataFrame({'Vehicle_ID': []}, geometry=[], crs=crs)

    grouped = df.groupby('Vehicle_ID', sort=False)
    summary = pd.DataFrame(index=grouped.size().index)
    for col in ('Vehicle_Class', 'Drone_ID'):
        if col in df.columns:
            summary[col] = grouped[col].first()
    summary['Num_Points'] = grouped.size()
    time_col = next((c for c in ('Timestamp', 'Local_Time') if c in df.columns), None)
    if time_col:
        times = df[time_col].where(df[time_col].astype(str) != UNDEFINED_TIMESTAMP)
        summary['Start_Time'] = times.groupby(df['Vehicle_ID'], sort=False).first()
        summary['End_Time'] = times.groupby(df['Vehicle_ID'], sort=False).last()
    if 'Frame_Number' in df.columns:
        summary['Start_Frame'] = grouped['Frame_Number'].min()
        summary['End_Frame'] = grouped['Frame_Number'].max()
    if 'Vehicle_Speed' in df.columns:
        speed = pd.to_numeric(df['Vehicle_Speed'], errors='coerce').groupby(df['Vehicle_ID'], sort=False)
        summary['Mean_Speed'] = speed.mean().round(1)
        summary['Max_Speed'] = speed.max().round(1)
    for col in ('Vehicle_Length', 'Vehicle_Width'):
        if col in df.columns:
            summary[col] = (
                pd.to_numeric(df[col], errors='coerce').groupby(df['Vehicle_ID'], sort=False).median().round(2)
            )

    indices = pd.factorize(df['Vehicle_ID'], sort=False)[0]
    lines = shapely.linestrings(df[[x_col, y_col]].to_numpy(dtype=float), indices=indices)
    return gpd.GeoDataFrame(summary.reset_index(), geometry=lines, crs=crs)


def output_path_for(csv_file: Path, args: argparse.Namespace) -> Path:
    """Return the export path for *csv_file*: '<stem>_<geometry>[_local]<ext>' in the chosen output folder.

    Without --output-folder the file goes next to its CSV. With it, a folder input keeps its
    sub-folder layout below the output folder, so same-named CSVs from different flights cannot
    overwrite each other; a single-file input is written into the output folder directly.
    """
    crs_suffix = '_local' if args.crs == 'local' else ''
    name = f"{csv_file.stem}_{args.geometry}{crs_suffix}{FORMATS[args.format][1]}"
    if args.output_folder is None:
        return csv_file.parent / name
    if args.input.is_dir():
        return args.output_folder / csv_file.parent.relative_to(args.input) / name
    return args.output_folder / name


def write_layer(gdf: gpd.GeoDataFrame, path: Path, fmt: str) -> None:
    """Write *gdf* to *path* atomically, as one layer named after the file stem.

    The layer name is passed explicitly for both drivers: GDAL otherwise derives it from the temporary
    file, and GeoJSON stores it in its top-level 'name' member.

    The temporary file keeps the real extension (``keep_suffix``), since GDAL warns when a GeoPackage
    is written under another one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = FORMATS[fmt][0]
    with atomic_output(path, keep_suffix=True) as tmp_path:
        gdf.to_file(tmp_path, driver=driver, layer=path.stem)


def resolve_crs(crs_mode: str, config: dict, logger: logging.Logger) -> str:
    """Return the CRS of the exported coordinates: EPSG:4326, or cfg -> georef -> transformation -> target_crs."""
    if crs_mode == 'wgs84':
        return WGS84
    target_crs = (config.get('georef', {}).get('transformation', {}) or {}).get('target_crs')
    if not target_crs:
        logger.critical("'--crs local' needs cfg -> georef -> transformation -> target_crs, which is not set.")
        sys.exit(1)
    return target_crs


def export_file(csv_file: Path, crs: str, args: argparse.Namespace, logger: logging.Logger) -> str:
    """Export one trajectory CSV and return its outcome: 'written', 'skipped' (nothing to export), or 'failed'."""
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as e:
        logger.error(f"Could not read '{csv_file}': {e}")
        return 'failed'
    df = drop_invalid_coordinates(df, args.crs, csv_file, logger)
    if df.empty:
        logger.warning(f"Skipped '{csv_file}': no rows with valid coordinates.")
        return 'skipped'

    if args.geometry == 'lines':
        gdf = build_lines(df, args.crs, crs, csv_file, logger)
    else:
        gdf = build_points(df, args.crs, crs)
    if gdf.empty:
        logger.warning(f"Skipped '{csv_file}': no vehicle has 2 or more valid points to form a line.")
        return 'skipped'

    path = output_path_for(csv_file, args)
    try:
        write_layer(gdf, path, args.format)
    except Exception as e:
        logger.error(f"Failed to write '{path}': {e}")
        return 'failed'
    logger.info(f"Exported {len(gdf)} {'vehicle' if args.geometry == 'lines' else 'point'} feature(s) to '{path}'.")
    return 'written'


def export_trajectories(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Export every trajectory CSV that *args.input* resolves to.

    Exits 1 when any file failed to be read or written, after attempting all of them, so a scripted
    run notices; a file with nothing to export (no valid coordinates, no line-forming vehicle) is
    skipped with a warning and does not count as a failure.
    """
    config = load_config(args.cfg, logger, args)
    crs = resolve_crs(args.crs, config, logger)
    if args.format == 'geojson' and args.crs == 'local':
        logger.warning(
            "GeoJSON (RFC 7946) expects WGS84 coordinates; some tools will misplace a local-CRS GeoJSON. "
            "Consider '--format gpkg' for the local CRS."
        )

    csv_files = find_input_csvs(args.input, args.crs, config.get('output', {}), logger)
    logger.info(f"Exporting {len(csv_files)} file(s) as {args.format} {args.geometry} in {crs}.")
    outcomes = [export_file(f, crs, args, logger) for f in csv_files]
    written, failed = outcomes.count('written'), outcomes.count('failed')
    logger.notice(f"Exported {written} of {len(csv_files)} file(s)" + (f"; {failed} failed." if failed else "."))
    if failed:
        sys.exit(1)


def parse_cli_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Export georeferenced trajectories to GIS formats (GeoPackage, GeoJSON)'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='A georeferenced .csv file, a video file (its georeferenced .csv is looked up in the output '
        'folder), or a folder searched recursively for georeferenced and aggregated .csv files.',
    )

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument(
        '--format',
        '-f',
        choices=list(FORMATS),
        default='gpkg',
        help='Output format: gpkg (GeoPackage) or geojson (default: gpkg).',
    )
    optional.add_argument(
        '--geometry',
        '-g',
        choices=['lines', 'points'],
        default='lines',
        help='lines = one LineString per vehicle with summary attributes; points = one Point per '
        'trajectory row with every CSV column (default: lines).',
    )
    optional.add_argument(
        '--crs',
        '-crs',
        choices=list(COORDINATE_COLUMNS),
        default='wgs84',
        help='wgs84 = Latitude/Longitude in EPSG:4326; local = Local_X/Local_Y in cfg -> georef -> '
        'transformation -> target_crs (default: wgs84).',
    )
    optional.add_argument(
        '--output-folder',
        '-of',
        type=Path,
        default=None,
        help='Folder for the exported files. Defaults to the folder of each input .csv; for a folder '
        'input, the sub-folder layout below it is mirrored.',
    )
    cfg_paths = add_common_args(optional, output_folder=False)

    return finalize_cli_args(parser, cfg_paths, argv)


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(__name__, args.verbose, args.log_path)

    export_trajectories(args, logger)


if __name__ == '__main__':
    main()
