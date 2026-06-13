# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared CLI argument helpers used by the pipeline entry points."""

from pathlib import Path

DEFAULT_CFG = 'geotrax/cfg/default.yaml'


def add_common_args(group, cfg: bool = True) -> None:
    """
    Register the options shared by all geo-trax commands on the given parser/group:
    ``--cfg`` (unless ``cfg=False``), ``--log-file``, and ``--verbose``.
    """
    if cfg:
        group.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG, help='Path to the main geo-trax configuration file.')
    group.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    group.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING).')
