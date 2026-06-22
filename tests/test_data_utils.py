# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the visualization and plotting color palettes."""

import re

from geotrax.utils.data_utils import PlotColors, VizColors


def test_vizcolors_hex2rgb():
    assert VizColors.hex2rgb('#1F77B4') == (31, 119, 180)
    assert VizColors.hex2rgb('#000000') == (0, 0, 0)
    assert VizColors.hex2rgb('#FFFFFF') == (255, 255, 255)


def test_vizcolors_call_rgb_and_bgr():
    colors = VizColors()
    assert colors(0) == (78, 121, 167)
    assert colors(0, bgr=True) == (167, 121, 78)


def test_vizcolors_index_wraps_around():
    colors = VizColors()
    assert colors(colors.n) == colors(0)
    assert colors(colors.n + 3) == colors(3)
    assert colors.txt_color == (255, 255, 255)


def test_plotcolors_returns_configured_color():
    palette = PlotColors(['#aaaaaa', '#bbbbbb'])
    assert palette.get_color(0) == '#aaaaaa'
    assert palette.get_color(1) == '#bbbbbb'


def test_plotcolors_falls_back_to_random_hex():
    palette = PlotColors(['#aaaaaa'])
    fallback = palette.get_color(5)
    assert re.fullmatch(r'#[0-9a-f]{6}', fallback)


def test_plotcolors_set_colors():
    palette = PlotColors()
    palette.set_colors(['#123456'])
    assert palette.get_color(0) == '#123456'
