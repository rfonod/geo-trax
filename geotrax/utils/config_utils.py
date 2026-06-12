# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""YAML configuration loading and path resolution for the geo-trax pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import yaml

from geotrax import CFG_DIR, PACKAGE_DIR

ROOT_DIR = PACKAGE_DIR.parent  # repository root (source checkout) or site-packages (installed)


def resolve_config_path(cfg_filepath: Union[str, Path]) -> Path:
    """Resolve a configuration file path.

    Tries, in order: the path as given (absolute or relative to the current working directory),
    the path relative to the package parent directory, and the path inside the bundled
    configuration directory (geotrax/cfg). A missing '.yaml' suffix and a legacy leading 'cfg/'
    component are tolerated, so e.g. 'confident', 'cfg/default.yaml', and 'tracker/default_ocsort'
    all resolve to the bundled configs. Returns the path unchanged if no candidate exists.
    """
    path = Path(cfg_filepath)
    if not path.suffix:
        path = path.with_suffix('.yaml')

    candidates = [path]
    if not path.is_absolute():
        bundled = Path(*path.parts[1:]) if path.parts[0] == 'cfg' else path
        candidates += [ROOT_DIR / path, CFG_DIR / bundled]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(cfg_filepath)


def resolve_asset_path(filepath: Union[str, Path]) -> Path:
    """Resolve a non-config asset path (e.g., model weights) against the cwd and the package parent.

    Returns the path unchanged if no candidate exists, leaving error reporting to the caller.
    """
    path = Path(filepath)
    if not path.is_absolute() and not path.is_file() and (ROOT_DIR / path).is_file():
        return ROOT_DIR / path
    return path


def load_config_all(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Load all configuration files and return a nested dict."""
    kwargs_main = load_config(args.cfg, logger)
    kwargs_stabilo = load_config(kwargs_main['cfg_stabilo'], logger)
    kwargs_ultralytics = load_config(kwargs_main['cfg_ultralytics'], logger)
    kwargs_georef = load_config(kwargs_main['cfg_georef'], logger)

    kwargs_ultralytics['model'] = str(resolve_asset_path(kwargs_ultralytics['model']))
    class_names_filepath = Path(kwargs_ultralytics['model']).with_suffix('.yaml')
    kwargs_main['class_names'] = load_class_names(class_names_filepath, logger)
    kwargs_main['args'] = args

    keys_to_update = ['classes', 'conf', 'show']
    for arg, value in vars(args).items():
        if value is not None and arg in keys_to_update:
            kwargs_ultralytics[arg] = value
            logger.info(f"The default ultralytics value for {arg} has been updated to the provided CLI argument: {value}.")
    kwargs_ultralytics['tracker'] = str(resolve_config_path(kwargs_main['cfg_tracker']))

    logger.info(f"The main configuration file and all sub-configurations therein have been loaded from: '{args.cfg}'.")

    return {
        'main': kwargs_main,
        'stabilo': kwargs_stabilo,
        'ultralytics': kwargs_ultralytics,
        'georef': kwargs_georef
    }


def load_config(cfg_filepath: Union[str, Path], logger: logging.Logger) -> dict:
    """Load a configuration file and return the contents as a dictionary."""
    resolved_filepath = resolve_config_path(cfg_filepath)
    try:
        with open(resolved_filepath, 'r') as f:
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
