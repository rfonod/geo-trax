# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import argparse
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Tuple, Union

import cv2
import yaml

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])


class bcolors:
    """
    Color palette for terminal output.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

NOTICE_LEVEL = 25
logging.addLevelName(NOTICE_LEVEL, "NOTICE")


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for colored log output
    """
    def format(self, record):
        message = super().format(record)
        if record.levelno == NOTICE_LEVEL:
            message = f"{bcolors.OKCYAN}{message}{bcolors.ENDC}"
        elif record.levelno == logging.WARNING:
            message = f"{bcolors.WARNING}{message}{bcolors.ENDC}"
        elif record.levelno == logging.ERROR:
            message = f"{bcolors.FAIL}{message}{bcolors.ENDC}"
        elif record.levelno == logging.CRITICAL:
            message = f"{bcolors.FAIL}{bcolors.BOLD}{message}{bcolors.ENDC}"
        return message


class FileFormatter(logging.Formatter):
    """
    Custom formatter for log output to file
    """
    def format(self, record):
        message = super().format(record)
        for color in vars(bcolors).values():
            if isinstance(color, str):
                message = message.replace(color, '')
        return message


def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LEVEL):
        self._log(NOTICE_LEVEL, message, args, **kwargs)
logging.Logger.notice = notice


def setup_logger(name: str, verbose: bool = False, filename: str = '', dry_run: bool = False, log_dir: str = 'logs') -> logging.Logger:
    """
    Set up a logger with a given name, verbosity level, and optional log file name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_format = '%(asctime)s - %(levelname)s - %(name)s:%(module)s:%(funcName)s - %(message)s'
    colored_formatter = ColoredFormatter(log_format)
    file_formatter = FileFormatter(log_format)

    console_level = NOTICE_LEVEL if not verbose else logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    if not dry_run:
        log_filepath = Path(log_dir) / (filename or f"{name.split('.')[0]}.log")
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_filepath}")

    logger._original_formatters = {h: h.formatter for h in logger.handlers}

    return logger


def load_config_all(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """
    Load all configuration files and return the contents as dictionaries
    """

    kwargs_main = load_config(args.cfg, logger)
    kwargs_stabilo = load_config(kwargs_main['cfg_stabilo'], logger)
    kwargs_ultralytics = load_config(kwargs_main['cfg_ultralytics'], logger)
    kwargs_georef = load_config(kwargs_main['cfg_georef'], logger)

    class_names_filepath = Path(kwargs_ultralytics['model']).with_suffix('.yaml')
    kwargs_main['class_names'] = load_class_names(class_names_filepath, logger)
    kwargs_main['args'] = args

    keys_to_update = ['classes']
    for arg, value in vars(args).items():
        if value is not None and arg in keys_to_update:
            kwargs_ultralytics[arg] = value
            logger.info(f"The default ultralytics value for {arg} has been updated to the provided CLI argument: {value}.")
    kwargs_ultralytics['tracker'] = kwargs_main['cfg_tracker']

    config = {
        'main': kwargs_main,
        'stabilo': kwargs_stabilo,
        'ultralytics': kwargs_ultralytics,
        'georef': kwargs_georef
    }

    logger.info(f"The main configuration file and all sub-configurations therein have been loaded from: '{args.cfg}'.")

    return config


def load_config(cfg_filepath: Path, logger: logging.Logger) -> dict:
    """
    Load a configuration file and return the contents as a dictionary
    """
    try:
        with open(cfg_filepath, 'r') as f:
            kwargs = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file '{cfg_filepath}' not found.")
        sys.exit(1)
    return kwargs


def load_class_names(class_names_filepath: Path, logger: logging.Logger) -> dict:
    """
    Load class names from a YAML file
    """
    try:
        with open(class_names_filepath, 'r') as f:
            class_names = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Class names '{class_names_filepath}' not found. Using default class names.")
        class_names = {i: f'class_{i}' for i in range(100)}
    else:
        logger.info(f"Class names loaded from: '{class_names_filepath}'.")
    return class_names


def detect_delimiter(filepath: Path, lines_to_check: int = 5) -> str:
    """
    Detect the delimiter of a CSV file by reading a few lines
    """
    delimiters = {',': 0, ' ': 0, '\t': 0}
    with open(filepath, 'r') as file:
        for _ in range(lines_to_check):
            line = file.readline()
            if not line:
                break
            delimiters[','] += line.count(',')
            delimiters[' '] += line.count(' ')
            delimiters['\t'] += line.count('\t')

    return max(delimiters, key=lambda k: delimiters[k])


def convert_to_serializable(obj):
    """
    Convert an object to a serializable format
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, argparse.Namespace):
        return {k: convert_to_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

class VizColors:
    """
    Color palette for plotting.
    """
    def __init__(self):
        hexs = ('1F77B4', 'D62728', 'FF7F0E', '006400', '8C564B', '9467BD',
                '0000FF', 'FF0000', 'A52A2A', '000000', '00FF00', '800080')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.txt_color = (255, 255, 255)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class PlotColors:
    """
    Color palette for plotting.
    """
    def __init__(self, colors=None):
        self.colors = colors if colors else []

    def set_colors(self, colors):
        self.colors = colors

    def get_color(self, index: int) -> str:
        if index < len(self.colors):
            return self.colors[index]
        else:
            return "#{:06x}".format(random.randint(0, 0xFFFFFF)) # Return a random color


def determine_location_id(source: Path, logger: logging.Logger = None) -> str:
    """
    Extract the location ID from the source filename. Location ID is the first sequence
    of alphabetic characters in the filename. Symbols '_' and '-' can act as separators.
    Examples:
    'A1.mp4' -> 'A'
    '2025-01-01_A_PM1.mp4' -> 'A'
    'A1_AV.csv' -> 'A'
    """
    location_id = []
    for char in source.stem:
        if char.isalpha():
            location_id.append(char)
        elif len(location_id) and (char in '_-' or char.isdigit()):
            break
    location_id = ''.join(location_id)

    if not location_id:
        message = f"Error: Failed to extract location ID from the source filename {source}."
        if logger:
            logger.error(message)
        else:
            print(message)
        sys.exit(1)

    if logger:
        logger.info(f"Detected location ID: '{location_id}' from the source filename {source.name}.")

    return location_id


def get_ortho_folder(source: Path, ortho_folder: Union[Path, None], logger: logging.Logger, critical = True) -> Path:
    """
    Get the orthophoto folder from the provided path or use the default folder structure.
    """
    if ortho_folder is None:
        ortho_folder = source.parent

        while ortho_folder != ortho_folder.parent:
            if ortho_folder.name in ['PROCESSED', 'DATASET']:
                break
            ortho_folder = ortho_folder.parent

        if ortho_folder.name not in ['PROCESSED', 'DATASET']:
            if critical:
                logger.critical(
                    f"Failed to find the orthophoto folder for source '{source}'. "
                    f"Please either provide a custom path using the --ortho-folder argument, "
                    f"disable georeferencing with the --no-geo (or -ng) flag, "
                    f"or ensure that the default folder structure is in place."
                )
                sys.exit(1)
            else:
                logger.info(
                    f"Failed to find the orthophoto folder for source '{source}'. "
                    f"Please either provide a custom path using the --ortho-folder argument, "
                    f"disable georeferencing with the --no-geo (or -ng) flag, "
                    f"or ensure that the default folder structure is in place."
                )
                return None

        ortho_folder = ortho_folder.parent / 'ORTHOPHOTOS'

    if not ortho_folder.exists():
        if critical:
            logger.critical(f"Orthophoto folder '{ortho_folder}' not found. Use the '--ortho-folder' argument to provide a custom path or ensure the default folder structure.")
            sys.exit(1)
        else:
            logger.info(f"Orthophoto folder '{ortho_folder}' not found. Use the '--ortho-folder' argument to provide a custom path or ensure the default folder structure.")
            return None
    else:
        logger.info(f"Using orthophoto folder: '{ortho_folder}'.")

    return ortho_folder


def determine_suffix_and_fourcc() -> Tuple[str, str]:
    """
    Determine the suffix and fourcc for the output video format.
    """
    suffix = 'mp4' if MACOS else 'avi' if WINDOWS else 'mp4'
    fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'
    return suffix, fourcc


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """
    Get the width and height of the video.
    """
    reader = cv2.VideoCapture(str(video_path))
    w_I = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_I = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.release()
    return w_I, h_I


def check_if_results_exist(file: Path, result_type: str, viz_mode: int = None, ext: str = None) -> Tuple[bool, Path]:
    """
    Check if the results already exist for the video file.
    """
    results_dir = file.parent / 'results'
    result_path = {
        "video": file,
        "processed": results_dir / f"{file.stem}.txt",
        'video_transformations': results_dir / f"{file.stem}_vid_transf.txt",
        'geo_transformations': results_dir / f"{file.stem}_geo_transf.txt",
        "georeferenced": results_dir / f"{file.stem}.csv",
        "visualized": results_dir / f"{file.stem}_mode_{viz_mode}.{ext}"
    }.get(result_type)
    return result_path.exists() if result_path else False, result_path
