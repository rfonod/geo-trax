#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Robert Fonod (robert.fonod@ieee.org)

"""
registration.py - Image registration via Stabilo.

Estimates a homography between two images by delegating feature detection, matching and
robust model fitting to a `stabilo` Stabilizer. Every stabilo parameter is configurable:
keyword arguments are forwarded verbatim on top of DEFAULT_STABILIZER_KWARGS, so new
stabilo options (such as the learning-based detectors and their `device`) need adding only
to the config, mirroring how the extraction stage does `Stabilizer(**config['stabilo'])`.
Only the registration geometry in FIXED_STABILIZER_KWARGS is pinned. This is the single
implementation shared by the georeferencing stage and the analysis tools.
"""

import logging

import cv2
import numpy as np
from stabilo import Stabilizer

# Registration defaults: the single source of truth, and the fallback for any key absent from
# cfg -> georef -> matching. Every key here is overridable from the config or the CLI.
DEFAULT_STABILIZER_KWARGS = {
    'detector_name': 'rsift',
    'matcher_name': 'bf',
    'filter_type': 'ratio',
    'sift_enable_precise_upscale': True,
    'rsift_eps': 1e-8,
    'max_features': 250000,
    'filter_ratio': 0.55,
    'ransac_method': cv2.USAC_MAGSAC,
    'ransac_epipolar_threshold': 3.0,
    'ransac_max_iter': 10000,
    'ransac_confidence': 0.999999,
    'clahe': False,
    'downsample_ratio': 1.0,
    'gpu': False,
    'gpu_device_id': 0,
}

# Registration geometry pinned by geo-trax regardless of config: callers assume a
# full-resolution 3x3 src->dst homography estimated from a single image pair with no
# exclusion mask. A config value for any of these is ignored (with a warning).
FIXED_STABILIZER_KWARGS = {
    'transformation_type': 'projective',
    'mask_use': False,
    'ref_multiplier': 1.0,
    'match_query_frame': 'current',
}

MIN_MAX_FEATURES = 10000   # do not retry below this feature count
DL_MAX_FEATURES_CAP = 8192  # ceiling for the learning-based detectors, which take max_features as top_k

# Derived from stabilo so the CLI choices cannot drift when it gains a detector or device.
DETECTOR_CHOICES = list(Stabilizer.VALID_DETECTORS)
DEVICE_CHOICES = list(Stabilizer.VALID_DEVICES)
DL_DETECTORS = set(Stabilizer.DL_DETECTORS)

# Every key estimate_homography knowingly forwards to the Stabilizer (mirrors cfg -> georef -> matching
# in cfg/default.yaml). Update alongside that config block when a new stabilo option is adopted.
KNOWN_STABILIZER_KWARGS = (
    set(DEFAULT_STABILIZER_KWARGS)
    | set(FIXED_STABILIZER_KWARGS)
    | {'device', 'loftr_weights', 'loftr_confidence', 'disk_weights', 'dedode_detector_weights', 'dedode_descriptor_weights'}
)


def _clamp_dl_max_features(kwargs: dict, logger: logging.Logger) -> None:
    """Cap max_features for the learning-based detectors, which consume it as top_k / num_features."""
    if kwargs['detector_name'] in DL_DETECTORS and kwargs['max_features'] > DL_MAX_FEATURES_CAP:
        logger.warning(
            f"detector_name='{kwargs['detector_name']}' is a learning-based detector; capping max_features "
            f"{kwargs['max_features']} -> {DL_MAX_FEATURES_CAP}. The registration default is sized for RootSIFT "
            f"and would exhaust memory here. Set cfg -> georef -> matching -> max_features <= "
            f"{DL_MAX_FEATURES_CAP} to silence this."
        )
        kwargs['max_features'] = DL_MAX_FEATURES_CAP


def estimate_homography(img_src: np.ndarray, img_dst: np.ndarray, logger: logging.Logger,
                        **stabilizer_kwargs) -> tuple:
    """
    Estimate the homography H mapping source -> destination image coordinates.

    The destination is set as the Stabilizer's reference frame and the source as the
    current frame, so the resulting cur->ref transform maps src -> dst with the RANSAC
    reprojection threshold evaluated in destination coordinates.

    `stabilizer_kwargs` is forwarded verbatim to the Stabilizer on top of
    DEFAULT_STABILIZER_KWARGS, so any parameter stabilo accepts (including the
    learning-based detectors' `device` and weight settings) can be driven from
    cfg -> georef -> matching without a signature change here. The registration geometry in
    FIXED_STABILIZER_KWARGS is pinned and overrides any supplied value.

    Set `gpu=True` (with `gpu_device_id` selecting the CUDA device) to CUDA-accelerate the
    registration. stabilo only GPU-accelerates the ORB detector, so `gpu=True` requires
    `detector_name='orb'` and a CUDA-enabled OpenCV build; otherwise stabilo raises ValueError
    (there is no CPU fallback). For the learning-based detectors use `device='cuda'` instead.

    If detection or matching fails, `max_features` is halved and retried, down to
    MIN_MAX_FEATURES. The detector-free `loftr` ignores max_features, so it is never retried.

    Returns:
        (H, inliers_count, num_matches, (n_src_kpts, n_dst_kpts)) on success, or
        (None, None, None, None) on failure. `inliers_count` is the number of RANSAC inliers
        and `num_matches` is the number of good matches fed to findHomography (i.e. the
        'inliers_count out of num_matches matches' figures).
    """
    unknown = sorted(set(stabilizer_kwargs) - KNOWN_STABILIZER_KWARGS)
    if unknown:
        logger.warning(f"Ignoring unrecognized registration key(s) {unknown}; check cfg -> georef -> matching for typos.")

    kwargs = {**DEFAULT_STABILIZER_KWARGS, **stabilizer_kwargs}

    clashes = sorted(
        k for k, v in FIXED_STABILIZER_KWARGS.items() if k in stabilizer_kwargs and stabilizer_kwargs[k] != v
    )
    if clashes:
        pinned = {k: FIXED_STABILIZER_KWARGS[k] for k in clashes}
        logger.warning(f"Ignoring registration-fixed key(s) {clashes}; they are pinned to {pinned}.")
    kwargs.update(FIXED_STABILIZER_KWARGS)

    _clamp_dl_max_features(kwargs, logger)
    detector_name = kwargs['detector_name']
    max_features_to_try = kwargs.pop('max_features')

    while True:
        stabilizer = Stabilizer(max_features=max_features_to_try, **kwargs)
        stabilizer.set_ref_frame(img_dst)
        stabilizer.stabilize(img_src)
        homography = stabilizer.get_cur_trans_matrix()

        if homography is not None:
            n_dst_kpts, n_src_kpts = stabilizer.get_cur_num_keypoints()  # (ref=dst, cur=src)
            inliers_count = stabilizer.get_cur_inliers_count()
            num_matches = stabilizer.get_cur_num_matches()
            return homography, inliers_count, num_matches, (n_src_kpts, n_dst_kpts)

        # loftr is detector-free, so halving max_features cannot change its outcome.
        next_try = max_features_to_try // 2
        if detector_name == 'loftr' or next_try < MIN_MAX_FEATURES:
            break
        logger.warning(
            f"Feature detection or matching failed with {max_features_to_try} max_features. "
            f"Trying with {next_try} max_features."
        )
        max_features_to_try = next_try

    logger.error("Feature detection failed with all attempted feature counts.")
    return None, None, None, None
