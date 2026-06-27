# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Geo-trax: georeferenced vehicle trajectory extraction from drone imagery."""

from pathlib import Path

__version__ = "1.0.1"  # single source of truth; pyproject.toml reads this via [tool.setuptools.dynamic]

PACKAGE_DIR = Path(__file__).resolve().parent  # geotrax/ package directory
CFG_DIR = PACKAGE_DIR / "cfg"                  # bundled configuration files
