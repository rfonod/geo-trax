# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for file utilities."""

import argparse
from pathlib import Path

import pytest

from geotrax.utils.file_utils import (
    build_result_path,
    check_if_results_exist,
    convert_to_serializable,
    detect_delimiter,
    determine_location_id,
    determine_suffix_and_fourcc,
    get_output_dir,
)


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


@pytest.mark.parametrize(
    'filename, expected',
    [
        ('A1.mp4', 'A'),
        ('2025-01-01_A_PM1.mp4', 'A'),
        ('A1_AV.csv', 'A'),
        ('BC12_xyz.txt', 'BC'),
    ],
)
def test_determine_location_id(filename, expected):
    assert determine_location_id(Path(filename)) == expected


def test_determine_location_id_without_alpha_exits():
    with pytest.raises(SystemExit):
        determine_location_id(Path('123.mp4'))


def test_convert_to_serializable():
    src = Path('/tmp/v.mp4')
    args = argparse.Namespace(source=src, conf=0.5)
    result = convert_to_serializable({'args': args, 'paths': [Path('a'), Path('b')]})
    assert result == {
        'args': {'source': str(src), 'conf': 0.5},
        'paths': ['a', 'b'],
    }


def test_determine_suffix_and_fourcc():
    suffix, fourcc = determine_suffix_and_fourcc()
    assert suffix in {'mp4', 'avi'}
    assert isinstance(fourcc, str) and len(fourcc) == 4


# --- get_output_dir ----------------------------------------------------------

@pytest.mark.parametrize(
    'cfg, source, expected_suffix',
    [
        (None, Path('/data/video.mp4'), 'results'),           # default folder
        ({'folder': 'out'}, Path('/data/video.mp4'), 'out'),  # custom relative folder
    ],
)
def test_get_output_dir_relative(cfg, source, expected_suffix):
    result = get_output_dir(source, cfg)
    assert result == source.parent / expected_suffix


def test_get_output_dir_absolute_folder():
    source = Path('/data/video.mp4')
    cfg = {'folder': '/shared/output'}
    result = get_output_dir(source, cfg)
    assert result == Path('/shared/output')


# --- build_result_path -------------------------------------------------------

@pytest.mark.parametrize(
    'result_type, extra_kwargs, expected_name',
    [
        ('processed', {}, 'video.txt'),
        ('video_transformations', {}, 'video_vid_transf.txt'),
        ('geo_transformations', {}, 'video_geo_transf.txt'),
        ('georeferenced', {}, 'video.csv'),
        ('visualized', {'viz_mode': 1, 'ext': 'mp4'}, 'video_mode_1.mp4'),
    ],
)
def test_build_result_path_default_cfg(result_type, extra_kwargs, expected_name):
    source = Path('/data/video.mp4')
    result = build_result_path(source, result_type, **extra_kwargs)
    assert result == source.parent / 'results' / expected_name


def test_build_result_path_video_returns_source():
    source = Path('/data/video.mp4')
    assert build_result_path(source, 'video') == source


def test_build_result_path_unknown_type_returns_none():
    assert build_result_path(Path('/data/video.mp4'), 'unknown') is None
