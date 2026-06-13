# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared CLI argument helpers used by the pipeline entry points."""

from pathlib import Path

DEFAULT_CFG = 'geotrax/cfg/default.yaml'


def add_common_args(group, cfg: bool = True) -> None:
    """
    Register the options shared by all geo-trax commands on the given parser/group:
    ``--cfg`` (unless ``cfg=False``), ``--log-path``, and ``--verbose``.
    """
    if cfg:
        group.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG, help='Path to the main geo-trax configuration file.')
    group.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write detailed logs: a directory (the default per-stage <stage>.log name is used inside it) or a full file path. Defaults to a platform-specific log directory.')
    group.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING).')
