# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for file utilities."""

import pytest

from geotrax.utils.file_utils import check_if_results_exist, detect_delimiter


@pytest.mark.parametrize(
    'content, expected',
    [
        ('1,2,3\n4,5,6\n', ','),
        ('1 2 3\n4 5 6\n', ' '),
        ('1\t2\t3\n4\t5\t6\n', '\t'),
    ],
)
def test_detect_delimiter(tmp_path, content, expected):
    filepath = tmp_path / 'results.txt'
    filepath.write_text(content)
    assert detect_delimiter(filepath) == expected


def test_check_if_results_exist(tmp_path):
    video = tmp_path / 'video.mp4'
    results = tmp_path / 'results'
    results.mkdir()

    exists, path = check_if_results_exist(video, 'processed')
    assert not exists and path == results / 'video.txt'

    (results / 'video.txt').touch()
    exists, _ = check_if_results_exist(video, 'processed')
    assert exists

    exists, path = check_if_results_exist(video, 'visualized', viz_mode=1, ext='mp4')
    assert not exists and path == results / 'video_mode_1.mp4'
