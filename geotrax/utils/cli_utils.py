# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared CLI argument helpers used by the pipeline entry points."""

from pathlib import Path

# Default pipeline config: the bundled 'default' preset. Resolved by resolve_config_path(),
# which falls back to the package's bundled cfg/ dir, so this works from any working directory
# and for both source checkouts and pip-installed wheels.
DEFAULT_CFG = 'geotrax/cfg/default.yaml'


def add_common_args(group, cfg: bool = True, output_folder: bool = True) -> None:
    """
    Register the options shared by all geo-trax commands on the given parser/group:
    ``--cfg`` (unless ``cfg=False``), ``--output-folder`` (unless ``output_folder=False``),
    ``--log-path``, and ``--verbose``.
    """
    if cfg:
        group.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG,
                           help="Pipeline config: a bundled preset name (default, confident, lenient, stable) or a path "
                                "to a custom config file. Run 'geotrax config show' to list presets or 'geotrax config copy' to customize.")
    if output_folder:
        group.add_argument('--output-folder', '-of', type=str, default=None,
                           help="Root folder for pipeline outputs. A bare name (e.g. 'results') creates a sub-folder next to "
                                "each input video; an absolute path is used as-is for all inputs in a batch "
                                "(not recommended when video filenames are not unique across the batch — outputs may overwrite each other). "
                                "Also sets the base under which plots/ is created. "
                                "Defaults to cfg -> output -> folder (historical default: 'results').")
    group.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write detailed logs: a directory (the default per-stage <stage>.log name is used inside it) or a full file path. Defaults to a platform-specific log directory.')
    group.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING).')
