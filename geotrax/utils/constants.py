# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared constants: platform flags, recognized file formats, data-quality thresholds, and pipeline fallbacks."""

import platform

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}
RESULTS_FORMATS = {'.txt', '.csv'}

# Plausibility thresholds: trajectory points exceeding these are reported as likely outliers
ACC_THRESHOLD_ALERT = 5    # acceleration magnitude [m/s^2]
SPEED_THRESHOLD_ALERT = 90  # speed [km/h]

# Fallback when the active tracker's config block omits a value
DEFAULT_TRACK_BUFFER = 30  # frames
