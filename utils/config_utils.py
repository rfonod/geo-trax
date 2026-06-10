# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""YAML configuration loading for the geo-trax pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml


def load_config_all(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Load all configuration files and return a nested dict."""
    kwargs_main = load_config(args.cfg, logger)
    kwargs_stabilo = load_config(kwargs_main['cfg_stabilo'], logger)
    kwargs_ultralytics = load_config(kwargs_main['cfg_ultralytics'], logger)
    kwargs_georef = load_config(kwargs_main['cfg_georef'], logger)

    class_names_filepath = Path(kwargs_ultralytics['model']).with_suffix('.yaml')
    kwargs_main['class_names'] = load_class_names(class_names_filepath, logger)
    kwargs_main['args'] = args

    keys_to_update = ['classes', 'conf', 'show']
    for arg, value in vars(args).items():
        if value is not None and arg in keys_to_update:
            kwargs_ultralytics[arg] = value
            logger.info(f"The default ultralytics value for {arg} has been updated to the provided CLI argument: {value}.")
    kwargs_ultralytics['tracker'] = kwargs_main['cfg_tracker']

    logger.info(f"The main configuration file and all sub-configurations therein have been loaded from: '{args.cfg}'.")

    return {
        'main': kwargs_main,
        'stabilo': kwargs_stabilo,
        'ultralytics': kwargs_ultralytics,
        'georef': kwargs_georef
    }


def load_config(cfg_filepath: Path, logger: logging.Logger) -> dict:
    """Load a configuration file and return the contents as a dictionary."""
    try:
        with open(cfg_filepath, 'r') as f:
            kwargs = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file '{cfg_filepath}' not found.")
        sys.exit(1)
    return kwargs


def load_class_names(class_names_filepath: Path, logger: logging.Logger) -> dict:
    """Load class names from a YAML file."""
    try:
        with open(class_names_filepath, 'r') as f:
            class_names = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Class names '{class_names_filepath}' not found. Using default class names.")
        class_names = {i: f'class_{i}' for i in range(100)}
    else:
        logger.info(f"Class names loaded from: '{class_names_filepath}'.")
    return class_names
