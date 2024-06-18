# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict
import argparse

import yaml


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


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for colored log output
    """
    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.WARNING:
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


def setup_logger(name: str, verbose: bool = False, log_file: str = '') -> logging.Logger:
    """
    Set up a logger with a given name, verbosity level, and optional log file
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set logger level to INFO for all cases

    colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = FileFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_level = logging.INFO if verbose else logging.WARNING
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)        
        file_handler.setLevel(logging.INFO)  # Set file handler to log INFO and above
        logger.addHandler(file_handler)
        logger.info(f"[{name}] Logging to file: {log_file}.")

    return logger


def load_config_all(args: argparse.Namespace, logger: logging.Logger, script_caller: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load all configuration files and return the contents as dictionaries
    """
    
    # Load the main config file for geo-trax
    kwargs_main = load_config(args.cfg, logger, script_caller)
    kwargs_main['args'] = args

    # Load the stabilo config file
    kwargs_stabilo = load_config(kwargs_main['cfg_stabilo'], logger, script_caller)

    # Load the ultralytics config file
    kwargs_ultralytics = load_config(kwargs_main['cfg_ultralytics'], logger, script_caller)

    # Load the class names and update the kwargs
    model_filepath = kwargs_ultralytics['model']
    class_names_filepath = Path(model_filepath).with_suffix('.yaml')
    kwargs_main['class_names'] = load_class_names(class_names_filepath, logger, script_caller)

    # Update ultralytics kwargs with args and kwargs
    keys_to_update = ['classes']
    for arg, value in vars(args).items():
        if value is not None and arg in keys_to_update:
            kwargs_ultralytics[arg] = value
            logger.info(f"{script_caller} The default value for {arg} has been updated to {value}.")
    kwargs_ultralytics['tracker'] = kwargs_main['cfg_tracker']

    config = {
        'main': kwargs_main,
        'stabilo': kwargs_stabilo,
        'ultralytics': kwargs_ultralytics
    }

    return config


def load_config(cfg_filepath: str, logger: logging.Logger, script_caller: str) -> dict:
    """
    Load a configuration file and return the contents as a dictionary
    """
    try:
        cfg_filepath = Path(cfg_filepath).with_suffix('.yaml')
        with open(cfg_filepath, 'r') as f:
            kwargs = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"{script_caller} Configuration file '{cfg_filepath}' not found.")
        sys.exit(1)
    else:
        logger.info(f"{script_caller} Configuration file '{cfg_filepath}' loaded successfully.")
    return kwargs


def load_class_names(class_names_filepath: Path, logger: logging.Logger, script_caller: str) -> dict:
    """
    Load class names from a YAML file and return them as a dictionary
    """
    try:
        with open(class_names_filepath, 'r') as f:
            class_names = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"{script_caller} Class names '{class_names_filepath}' not found. Using default class names.")
        class_names = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'}
    else:
        logger.info(f"{script_caller} Class names '{class_names_filepath}' loaded successfully.")
    return class_names


def detect_delimiter(filepath: str, lines_to_check: int = 5) -> str:
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
            
    return max(delimiters, key=delimiters.get)


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

class Colors:
    """Color palette for plotting."""
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
    

