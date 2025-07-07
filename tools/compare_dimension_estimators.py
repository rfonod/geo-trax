#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
compare_dimension_estimators.py - Vehicle Dimension Estimation Comparison

Compares old and new methods for estimating vehicle dimensions from tracking data.
The old method uses area percentiles and bbox ratio filtering, while the new method
employs azimuth-based trajectory analysis for improved accuracy.

Usage:
  python tools/compare_dimension_estimators.py <source> [options]

Arguments:
  source : Path to video file or YAML configuration file.

Options:
  -p, --plot         : Plot length/width histograms for detailed analysis (default: False).
  -i, --id <int>     : Vehicle ID to analyze in detail (default: 0).
  -l, --lines <int>  : Number of output lines to display (default: 5).

Examples:
1. Compare methods for video:
   python tools/compare_dimension_estimators.py video.mp4

2. Detailed analysis with plots:
   python tools/compare_dimension_estimators.py video.yaml --plot --id 42

Input:
- Video file or YAML configuration with tracking results in results/ subdirectory
- Tracking data format: frame, ID, bbox coordinates, class information

Output:
- Timing comparison between old and new methods
- Statistical analysis of dimension differences (max, mean, median)
- Track IDs with maximum differences for further investigation
- Optional histogram plots for specified vehicle IDs

Notes:
- Requires pre-computed tracking results in results/ subdirectory
- New method uses azimuth-based filtering with configurable thresholds
- Old method relies on area percentiles and bbox ratio thresholds
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

GSD_150 = 0.0282  # meters per pixel at 3840x2160 resolution and 150m altitude
GSD_140 = 0.0263  # meters per pixel at 3840x2160 resolution and 140m altitude
GSD = (GSD_150 + GSD_140) / 2

tau_c = {
    0: 1.83,  # car (vans, SUVs, etc.)
    1: 2.85,  # bus
    2: 1.7,  # truck
    3: 1.8,  # motorcycle
    -1: 1.7,
}  # unknown

VIDEO_FORMATS = ['.mp4', '.MP4', '.mov', '.MOV']


def main(args):
    # Load tracks associated with the video (need to run tracking first)
    tracks_txt_file = Path(f"{str(args.source.parent / 'results' / args.source.stem)}.txt")
    tracks = np.loadtxt(tracks_txt_file, delimiter=' ')

    # Estimate vehicle dimensions with the old method
    start_time = time.time()
    tracks_old = estimate_vehicle_dimensions_old(tracks)
    time_old = time.time() - start_time

    # Estimate vehicle dimensions with the new method
    start_time = time.time()
    tracks_new = estimate_vehicle_dimensions(tracks, args)
    time_new = time.time() - start_time

    # print the time taken for each method
    if not args.plot:
        print(f"Computing with the OLD method took: {time_old} seconds")
        print(f"Computing with the NEW method took: {time_new} seconds")

    # compare the results
    analyse_results(tracks_new, tracks_old, args)


def analyse_results(tracks_new, tracks_old, args):
    df_new = pd.DataFrame(tracks_new)
    df_old = pd.DataFrame(tracks_old)

    print(df_new.head(args.lines))
    print(df_old.head(args.lines))

    # calculate the difference between the two methods, use only the last two columns
    diff_L = np.abs(tracks_new[:, -2] - tracks_old[:, -2])
    diff_W = np.abs(tracks_new[:, -1] - tracks_old[:, -1])

    # print the stats and round the results to 4 decimal places
    print(f"\fMax difference in length: {np.round(np.nanmax(diff_L), 4)}")
    print(f"Max difference in width: {np.round(np.nanmax(diff_W), 4)}")
    print(f"Mean difference in length: {np.round(np.nanmean(diff_L), 4)}")
    print(f"Mean difference in width: {np.round(np.nanmean(diff_W), 4)}")
    print(f"Median difference in length: {np.round(np.nanmedian(diff_L), 4)}")
    print(f"Median difference in width: {np.round(np.nanmedian(diff_W), 4)}")

    # print the id of the track with the maximum difference
    idx_L_max = np.nanargmax(diff_L)
    idx_W_max = np.nanargmax(diff_W)
    id_L_max = int(tracks_new[idx_L_max, 1])
    id_W_max = int(tracks_new[idx_W_max, 1])
    print(f"\nTrack ID with max difference in length: {id_L_max}")
    print(f"Track ID with max difference in width: {id_W_max}")

    # print the worst case scenarios side by side
    print(f'\nID:         {id_L_max:<20}{id_W_max:<20}')
    print(f'Old (L):    {tracks_old[idx_L_max][-2]:<20.4f}{tracks_old[idx_W_max][-1]:<20.4f}')
    print(f'New (L):    {tracks_new[idx_L_max][-2]:<20.4f}{tracks_new[idx_W_max][-1]:<20.4f}')
    print(f'Old (W):    {tracks_old[idx_L_max][-1]:<20.4f}{tracks_old[idx_W_max][-1]:<20.4f}')
    print(f'New (W):    {tracks_new[idx_L_max][-1]:<20.4f}{tracks_new[idx_W_max][-1]:<20.4f}')


def estimate_vehicle_dimensions(tracks, args, eps=4, r0=1.25, GSD=GSD, theta_bar_deg=15, tau_c=tau_c):
    # determine the radius for the azimuth-based filtering
    r = r0 / GSD

    # determine the angle threshold for the azimuth-based filtering
    theta_bar = np.deg2rad(theta_bar_deg)

    # get video resolution
    if args.source.suffix == '.yaml':
        with open(args.source, 'r') as f:
            video_info = yaml.load(f, Loader=yaml.FullLoader)
        w_I, h_I = video_info['main']['video']['w_I'], video_info['main']['video']['h_I']
    elif args.source.suffix in VIDEO_FORMATS:
        cap = cv2.VideoCapture(str(args.source))
        w_I, h_I = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    # Step 0: filter tracks with id != -1 directly in numpy array
    valid_tracks = tracks[tracks[:, 1] != -1]

    # Step 1: visibility filtering
    mask = (valid_tracks[:, 2] - valid_tracks[:, 4] / 2 > eps) & (valid_tracks[:, 3] - valid_tracks[:, 5] / 2 > eps)
    mask &= (valid_tracks[:, 2] + valid_tracks[:, 4] / 2 < w_I - 1 - eps) & (
        valid_tracks[:, 3] + valid_tracks[:, 5] / 2 < h_I - 1 - eps
    )
    valid_tracks = valid_tracks[mask]

    # Step 2: initial dimensions computation
    unique_ids = np.unique(valid_tracks[:, 1]).astype(int)
    id2lengths = {id: [] for id in unique_ids}
    id2widths = {id: [] for id in unique_ids}
    id2x_centers = {id: [] for id in unique_ids}
    id2y_centers = {id: [] for id in unique_ids}
    id2class = {}

    if valid_tracks.shape[1] > 8:
        idx_x, idx_y, idx_c = 6, 7, 10  # stabilized tracks available
    else:
        idx_x, idx_y, idx_c = 2, 3, 6  # consider the unstabilzed tracks if stabilization is not performed

    for track in valid_tracks:
        id = int(track[1])
        w, h = track[4], track[5]
        x_center, y_center = track[idx_x], track[idx_y]
        v_class = int(track[idx_c])
        id2lengths[id].append(max(w, h))
        id2widths[id].append(min(w, h))
        id2x_centers[id].append(x_center)
        id2y_centers[id].append(y_center)
        id2class.setdefault(id, v_class)

    # plot the distribution of vehicle lengths and widths
    plot_histograms(id2lengths, id2widths, args, 'Step 2')

    # Step 3: azimuth-based filtering
    for id in unique_ids:
        # get the dimensions and centers
        lengths = id2lengths[id]
        widths = id2widths[id]
        x_centers = id2x_centers[id]
        y_centers = id2y_centers[id]

        # calculate the azimuths based on the vehicle trajectory satisfying a distance threshold
        azimuth = None
        idx_prev = 0
        x_c_prev, y_c_prev = x_centers[idx_prev], y_centers[idx_prev]
        mask = np.zeros(len(lengths), dtype=bool)
        for idx, point in enumerate(zip(x_centers[1:], y_centers[1:]), start=1):
            x_c, y_c = point
            distance = np.sqrt((x_c - x_c_prev) ** 2 + (y_c - y_c_prev) ** 2)
            if distance >= r:
                azimuth = np.arctan2(-(y_c - y_c_prev), x_c - x_c_prev)
                x_c_prev, y_c_prev = x_c, y_c
                if np.any(np.abs(azimuth - np.array([0, np.pi / 2, np.pi, -np.pi / 2, -np.pi])) <= theta_bar):
                    mask[idx_prev:idx] = True

                # for debugging, print the azimuth-related information
                if args.id == id:
                    print(
                        idx_prev,
                        idx,
                        distance,
                        '>=',
                        r,
                        np.rad2deg(azimuth),
                        '|',
                        '\nL:',
                        np.array(lengths)[mask],
                        '\nW:',
                        np.array(widths)[mask],
                    )

                idx_prev = idx

        lengths, widths = np.array(lengths), np.array(widths)
        if azimuth is None:
            mask = lengths >= widths * tau_c[id2class.get(id, tau_c[-1])]  # ratio l/w > threshold (e.g., 1.5)
        id2lengths[id] = lengths[mask]
        id2widths[id] = widths[mask]

    # plot the distribution of vehicle lengths and widths
    plot_histograms(id2lengths, id2widths, args, 'Step 3')

    # Step 4: final dimension computation
    id2length, id2width = {}, {}

    for id in unique_ids:
        id2length[id] = np.percentile(id2lengths[id], 25) if len(id2lengths[id]) > 0 else np.nan
        id2width[id] = np.percentile(id2widths[id], 25) if len(id2widths[id]) > 0 else np.nan

        if args.id == id:
            print(f'\nID: {int(id)} | Length: {id2length[id]} | Width: {id2width[id]}\n')

    # Finally: append v_length and v_width to each track, per id, as two last columns
    tracks = np.append(tracks, np.zeros((len(tracks), 2)), axis=1)
    for i, track in enumerate(tracks):
        id = int(track[1])
        tracks[i, -2] = id2length.get(id, np.nan)
        tracks[i, -1] = id2width.get(id, np.nan)

    return tracks


def estimate_vehicle_dimensions_old(tracks, bbox_ratio_threshold=1.25, percentile_lower=25, percentile_upper=75):
    # filter tracks with id != -1 directly in numpy array
    valid_tracks = tracks[tracks[:, 1] != -1]

    # split valid_tracks into id groups
    unique_ids = np.unique(valid_tracks[:, 1])

    # initialize dictionaries to store lengths and widths
    id2v_lengths = {id_: [] for id_ in unique_ids}
    id2v_widths = {id_: [] for id_ in unique_ids}

    # store lengths and widths for each id
    for track in valid_tracks:
        id_ = int(track[1])
        w, h = track[4], track[5]
        id2v_lengths[id_].append(max(w, h))
        id2v_widths[id_].append(min(w, h))

    id2v_length = {}
    id2v_width = {}

    for id_ in unique_ids:
        v_lengths = id2v_lengths[id_]
        v_widths = id2v_widths[id_]

        # calculate v_areas within the loop
        v_areas = np.array(v_lengths) * np.array(v_widths)

        # filter out lengths and widths based on area percentiles
        percentile_lower_value = np.percentile(v_areas, percentile_lower)
        percentile_upper_value = np.percentile(v_areas, percentile_upper)
        mask = (v_areas > percentile_lower_value) & (v_areas < percentile_upper_value)
        v_lengths = np.array(v_lengths)[mask]
        v_widths = np.array(v_widths)[mask]

        # filter out lengths and widths based on bbox ratio threshold
        mask = v_lengths > v_widths * bbox_ratio_threshold  # ratio l/w > threshold (e.g., 1.5)
        v_lengths = v_lengths[mask]
        v_widths = v_widths[mask]

        # calculate medians
        v_length = np.nanmedian(v_lengths) if len(v_lengths) > 0 else np.nan
        v_width = np.nanmedian(v_widths) if len(v_widths) > 0 else np.nan

        id2v_length[id_] = v_length
        id2v_width[id_] = v_width

    # append v_length and v_width to each track, per id, as two last columns
    tracks = np.append(tracks, np.zeros((len(tracks), 2)), axis=1)
    for i, track in enumerate(tracks):
        id_ = int(track[1])
        tracks[i, -2] = id2v_length.get(id_, np.nan)
        tracks[i, -1] = id2v_width.get(id_, np.nan)

    return tracks


def plot_histograms(id2lengths, id2widths, args, title):
    if args.plot:
        id = args.id
        if id == 0:
            while id <= 0:
                id = int(input("Enter a valid ID to plot: "))
                if id not in id2lengths:
                    print(f"ID {id} not found in the tracks")
            args.id = id

        plt.hist(id2lengths[id], bins=50, alpha=0.5, label='Vehicle Lengths')
        plt.hist(id2widths[id], bins=50, alpha=0.5, label='Vehicle Widths')
        plt.title(f'ID {id} - {title}')
        plt.legend(loc='upper right')
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare the old and new methods for estimating vehicle dimensions")
    parser.add_argument("source", type=Path, help="Specific video file (or YAML file) to process")
    parser.add_argument("--plot", "-p", action="store_true", help="plot histograms [default: False]")
    parser.add_argument("--id", "-i", type=int, default=0, help="ID to print/plot in detail")
    parser.add_argument("--lines", "-l", type=int, default=5, help="Number of lines to print")

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
