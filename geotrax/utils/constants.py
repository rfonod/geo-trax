# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared constants: platform flags, recognized file formats, and data-quality thresholds."""

import platform

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}
RESULTS_FORMATS = {'.txt', '.csv'}

# Plausibility thresholds: trajectory points exceeding these are reported as likely outliers
ACC_THRESHOLD_ALERT = 5    # acceleration magnitude [m/s^2]
SPEED_THRESHOLD_ALERT = 90  # speed [km/h]
