#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
recut_video_and_csv.py - Video and Flight Log Re-cutting Tool

This script re-cuts video files and their corresponding flight logs according to specified frame ranges.
It performs keyframe-aligned cutting to avoid re-encoding and maintains synchronization between video
and CSV flight data.

The tool reads cut specifications from a text file and applies them to both the video and its
associated flight log, adjusting frame numbers and timestamps accordingly.

Usage:
  python tools/recut_video_and_csv.py <input_video> <cuts> [options]

Arguments:
  input_video : Path to the input video file to be cut.
  cuts        : Path to the cuts specification file.

Options:
  -h, --help         : Show this help message and exit.
  -o, --output <path> : Output file path for the cut video and flight log (default: auto-generated).
  -d, --debug        : Run in debug mode with verification and detailed output (default: False).

Cut File Format:
  cut_start_frame_number, cut_end_frame_number, [rotation]

  Where:
  - cut_start_frame_number : Frame where the cut video should start
  - cut_end_frame_number   : Frame where the cut video should end (-1 for end of video)
  - rotation               : Optional counter-clockwise rotation in degrees (0, ±90, ±180, ±270)

Examples:
1. Basic video cutting:
   python tools/recut_video_and_csv.py video.MP4 cuts.txt

2. Cut with custom output path:
   python tools/recut_video_and_csv.py video.MP4 cuts.txt --output cut_video.MP4

3. Debug mode with verification:
   python tools/recut_video_and_csv.py video.MP4 cuts.txt --debug

Input:
- Video file (any format supported by ffmpeg)
- Corresponding CSV flight log with same filename but .CSV extension
- Cuts file containing single line with frame range and optional rotation

Output:
- Cut video file (with _cut suffix if no output path specified)
- Cut CSV flight log with adjusted frame numbers and timestamps
- Debug verification output (if debug mode enabled)

Notes:
- Cut start frame may be adjusted to the nearest keyframe to avoid re-encoding
- Flight log must exist in the same directory as the video file
- Frame numbers in the CSV are adjusted relative to the new cut start
- Rotation is applied at metadata level without re-encoding
- Debug mode verifies cut accuracy by comparing frame differences
"""

import argparse
import os
import platform
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd


def process_video_cutting(filepaths: Dict[str, Path], debug: bool = False) -> None:
    cuts = get_cuts(filepaths)
    cuts_adjusted = get_adjusted_cuts(cuts, filepaths, debug)
    cut_and_save_video(filepaths, cuts_adjusted, debug)
    cut_and_save_csv(filepaths, cuts_adjusted, debug)


def cut_and_save_video(filepaths: Dict[str, Path], cuts: Tuple[int, int, int], debug: bool = False) -> None:
    cut_start, cut_end, rotation = cuts
    input_video = str(filepaths["input_video"])
    output_video = str(filepaths["output_video"])

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # cmd = f'ffmpeg -y -i {input_video} -ss {cut_start/fps} -to {cut_end/fps} -c copy {output_video}' # next keyframe (i-frame)
    # cmd = f'ffmpeg -y -ss {cut_start/fps} -to {cut_end/fps} -i {input_video} -c copy {output_video}' # previous keyframe (i-frame)
    # cmd = f'ffmpeg -y -i {input_video} -ss {cut_start/fps} -to {cut_end/fps} -async 1 -strict -2 {output_video}' # exact frames cut, but with re-encoding
    cmd = f'ffmpeg -y -i {input_video}'
    if cut_start > 0:
        cmd += f' -ss {cut_start / fps}'
    if cut_end != -1:
        cmd += f' -to {cut_end / fps}'
    cmd += f' -c copy {output_video}'  # next keyframe (i-frame)
    if not debug:
        cmd += ' -v quiet'
    try:
        print(f"Running the following command: {cmd}")
        os.system(cmd)
    except:
        print('\033[91m' + f"Problem with cutting the video '{input_video}'" + '\033[0m')
    else:
        print(f"Cut video saved to '{output_video}'")
    if debug:
        verify_cut(filepaths, cut_start, cut_end, debug)

    # if needed, perform counter-clockwise rotation (metadata level rotation, no re-encoding needed)
    if int(rotation) != 0 and debug == False:
        temp_filepath = output_video.replace(".MP4", "_temp.MP4")
        os.rename(output_video, temp_filepath)
        os.system(
            f'ffmpeg -i {temp_filepath} -c copy -map_metadata 0 -metadata:s:v rotate="{rotation}" {output_video} -v quiet'
        )
        os.remove(temp_filepath)


def cut_and_save_csv(filepaths: Dict[str, Path], cuts: Tuple[int, int, int], debug: bool = False) -> None:
    cut_start = cuts[0]
    cut_end = cuts[1]
    if cut_end == -1:
        cap = cv2.VideoCapture(str(filepaths["input_video"]))
        cut_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
    input_csv = filepaths["input_csv"]
    output_csv = filepaths["output_csv"]

    df = pd.read_csv(input_csv)
    df = df[(df['frame'] >= cut_start) & (df['frame'] <= cut_end)]
    df['frame'] = df['frame'] - cut_start
    try:
        df.to_csv(output_csv, index=False)
    except:
        print('\033[91m' + f"Problem with saving the cut flight log to '{output_csv}'" + '\033[0m')
    else:
        print(f"Saved the cut flight log to '{output_csv}'")


def verify_cut(
    filepaths: Dict[str, Path], cut_start: int, cut_end: int, debug=False, verify_N_frames: int = 30
) -> None:
    input_video = str(filepaths["input_video"])
    output_video = str(filepaths["output_video"])

    if cut_end == -1:
        cap = cv2.VideoCapture(input_video)
        cut_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
    total_frame_count = cut_end - cut_start
    verify_every_nth_frame = total_frame_count // verify_N_frames
    input_frames_selected, cut_frames_selected = [], []

    try:
        cap = cv2.VideoCapture(input_video)
    except:
        print('\033[91m' + f"Problem with reading '{input_video}'" + '\033[0m')
    else:
        print("Total number of frames in the input video:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("FPS of the input video:", cap.get(cv2.CAP_PROP_FPS))
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i >= cut_start and (i - cut_start) % verify_every_nth_frame == 0:
                input_frames_selected.append(frame)
            i += 1
    finally:
        cap.release()

    try:
        cap = cv2.VideoCapture(output_video)
    except:
        print('\033[91m' + f"Problem with reading '{output_video}'" + '\033[0m')
    else:
        print("Total number of frames in the cut video:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("FPS of the cut video:", cap.get(cv2.CAP_PROP_FPS))
        i = 0
        while cap.isOpened() and i < cut_end:
            ret, frame = cap.read()
            if not ret:
                break
            if i % verify_every_nth_frame == 0:
                cut_frames_selected.append(frame)
            i += 1
    finally:
        cap.release()

    for i, imgs_i in enumerate(zip(input_frames_selected, cut_frames_selected)):
        img_input_i, img_cut_i = imgs_i
        if img_input_i is None or img_cut_i is None:
            print('\033[91m' + f"Problem with reading frame {i} from the input/cut video" + '\033[0m')
            break
        img_diff = cv2.absdiff(img_input_i, img_cut_i)
        img_diff_rmse = np.sqrt(np.mean(img_diff**2))
        print(
            f"({i}) Mean RMSE of the cut frame #: {i * verify_every_nth_frame} wrt input frame #: {cut_start + i * verify_every_nth_frame} is: {img_diff_rmse}"
        )
        if debug:
            cv2.imshow(f"image {i}", img_diff)
            cv2.waitKey(200)
            # k = cv2.waitKey(0) & 0xFF
            # if k == ord('q'):
            #     break
    if debug:
        cv2.destroyAllWindows()


def get_adjusted_cuts(
    cuts: Tuple[int, int, int], filepaths: Dict[str, Path], debug: bool = False
) -> Tuple[int, int, int]:
    input_video = str(filepaths['input_video'])
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if platform.system() == "Windows" or platform.system() == "Darwin":
        key_frames_retrieval_cmd = (
            "ffprobe -loglevel error -select_streams v:0 "
            "-show_entries packet=pts_time,flags -of "
            f"csv=print_section=0 {input_video}"
            " | awk -F',' '/K/ {print $1}'"
        )
    elif platform.system() == "Linux":
        key_frames_retrieval_cmd = (
            f"ffmpeg -i {input_video} -vf select='eq(pict_type\\,PICT_TYPE_I)',showinfo "
            "-vsync vfr -f null - -loglevel debug 2>&1 | "
            "awk '/pts_time/ {gsub(/.*pts_time:/, \"\"); gsub(/ .*/, \"\"); print;}'"
        )
    key_frames = os.popen(key_frames_retrieval_cmd).read().split()
    key_frames_arr = np.array([float(key_frame) for key_frame in key_frames])

    # find the closest key-frame (from right) for the provided cut_start
    cut_start = cuts[0]
    if cut_start == 0:
        print("No need to adjust the cut start frame.")
        return cuts
    key_frames_diffs = key_frames_arr - cut_start / fps
    i_key_frame_closest = np.where(key_frames_diffs >= 0, key_frames_diffs, np.inf).argmin()
    closest_key_frame = float(key_frames[i_key_frame_closest])
    cut_start_adjusted = round(closest_key_frame * fps)
    print(
        f"Adjusted to cut the input video from frame {cut_start_adjusted} to frame {cuts[1]} with rotation {cuts[2]}."
    )

    if debug:
        print("Key frames (in sec):", key_frames)
        print("Key frame diffs wrt cut_start:", key_frames_diffs)
        print("Closest (from right) key frame (in sec):", closest_key_frame)
        print(f"Original frame cut start ({cut_start}) adjusted to {cut_start_adjusted}")

    return (cut_start_adjusted, cuts[1], cuts[2])


def get_cuts(filepaths: Dict[str, Path]) -> Tuple[int, int, int]:
    try:
        with open(filepaths['cuts_txt'], 'r') as f:
            cuts = [line.rstrip().split(',') for line in f if line.strip()]
    except FileNotFoundError:
        print('\033[91m' + f"Problem with reading '{filepaths['cuts_txt']}'" + '\033[0m')
        exit(1)

    if len(cuts) == 0:
        print('\033[91m' + f"The file '{filepaths['cuts_txt']}' is empty!" + '\033[0m')
        exit(1)
    elif len(cuts) > 1:
        print('\033[91m' + f"The file '{filepaths['cuts_txt']}' contains more than one line!" + '\033[0m')
        exit(1)
    else:
        cuts = cuts[0]
        cut_start = int(cuts[0].strip())
        cut_end = int(cuts[1].strip())
        try:
            cut_video_rotate = int(cuts[2].strip())
        except:
            cut_video_rotate = 0

    print(
        f"Requested to cut the input video from frame {cut_start} to frame {cut_end} with rotation {cut_video_rotate}."
    )
    cuts = (cut_start, cut_end, cut_video_rotate)
    perform_sanity_checks(cuts, filepaths)
    return cuts


def perform_sanity_checks(cuts: Tuple[int, int, int], filepaths: Dict[str, Path]) -> None:
    cap = cv2.VideoCapture(str(filepaths["input_video"]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cut_start, cut_end, rotation = cuts
    assert cut_start >= 0, f"Error, 'cut_start' must be non-negative numbers in {filepaths['cuts_txt']}"
    if cut_end == -1:
        cut_end = frame_count - 1
    assert cut_start < cut_end, f"Error, 'cut_start' >= 'cut_end' in {filepaths['cuts_txt']}"
    assert cut_end <= frame_count - 1, (
        f"Error, 'cut_end' in {filepaths['cuts_txt']} is greater than the total number of frames in the input video"
    )
    assert rotation in {0, 90, 180, 270, -90, -180, -270}, (
        f"Error, invalid rotation specified in {filepaths['cuts_txt']}"
    )
    assert filepaths["input_csv"].exists(), (
        f"Error, the corresponding flight log '{filepaths['input_csv']}' does not exist"
    )


def get_cli_arguments():
    parser = argparse.ArgumentParser(description="Cut video and flight log according to specified frame ranges.")
    parser.add_argument('input_video', type=Path, help="File path to the input video")
    parser.add_argument('cuts', type=Path, help="File path to the cuts file")
    parser.add_argument('--output', '-o', type=Path, default=None, help="Output file path for the cut video and flight log")
    parser.add_argument('--debug', '-d', action='store_true', help="Run in debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()

    filepaths = {
        'cuts_txt': args.cuts,
        'input_video': args.input_video,
        'input_csv': args.input_video.with_suffix('.CSV'),
    }

    if args.output:
        filepaths['output_video'] = args.output
        filepaths['output_csv'] = args.output.with_suffix('.CSV')
    else:
        filepaths['output_video'] = args.input_video.with_name(f"{args.input_video.stem}_cut{args.input_video.suffix}")
        filepaths['output_csv'] = args.input_video.with_name(f"{args.input_video.stem}_cut.CSV")

    process_video_cutting(filepaths, args.debug)
