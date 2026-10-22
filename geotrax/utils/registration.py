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
from typing import Any

import cv2
import numpy as np
from stabilo import Stabilizer

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
    'ransac_threshold_space': 'full',
    'ransac_max_iter': 10000,
    'ransac_confidence': 0.999999,
    'clahe': False,
    'downsample_ratio': 1.0,
    'gpu': False,
    'gpu_device_id': 0,
}
"""Registration defaults: RootSIFT, brute-force matching with a ratio test, and MAGSAC++.

The single source of truth for registration, and the fallback for any key absent from
`georef.matching`. Every key can be overridden from the config, the CLI or the keyword arguments
of [`estimate_homography`][geotrax.utils.registration.estimate_homography].
"""

FIXED_STABILIZER_KWARGS = {
    'transformation_type': 'projective',
    'mask_use': False,
    'ref_multiplier': 1.0,
    'match_query_frame': 'current',
}
"""Registration geometry pinned regardless of the config.

Callers assume a full-resolution 3x3 source-to-destination homography estimated from a single
image pair with no exclusion mask, so a supplied value for any of these keys is ignored with a
warning.
"""

MIN_MAX_FEATURES = 10000
"""Lowest `max_features` that the halving retry in `estimate_homography` goes down to."""

DL_MAX_FEATURES_CAP = 8192
"""Ceiling on `max_features` for the learning-based detectors, which consume it as `top_k`."""

DETECTOR_CHOICES = list(Stabilizer.VALID_DETECTORS)
"""Every detector name stabilo accepts; the `choices` of `--stab-detector` and `--geo-detector`."""

DEVICE_CHOICES = list(Stabilizer.VALID_DEVICES)
"""Every torch device name stabilo accepts; the `choices` of `--stab-device` and `--geo-device`."""

DL_DETECTORS = set(Stabilizer.DL_DETECTORS)
"""The learning-based (kornia) detectors, which run on a torch device instead of OpenCV."""

KNOWN_STABILIZER_KWARGS = (
    set(DEFAULT_STABILIZER_KWARGS)
    | set(FIXED_STABILIZER_KWARGS)
    | {'device', 'loftr_weights', 'loftr_confidence', 'disk_weights', 'dedode_detector_weights', 'dedode_descriptor_weights'}
)
"""Every key `estimate_homography` forwards to the Stabilizer without a warning.

Mirrors `georef.matching` in `default.yaml`; extend it together with that block when a new stabilo
option is adopted.
"""


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
                        **stabilizer_kwargs: Any) -> tuple:
    """Estimate the homography that maps source image coordinates to destination coordinates.

    The destination is the Stabilizer's reference frame and the source its current frame, so the
    RANSAC reprojection threshold is evaluated in destination coordinates. If detection or matching
    fails, `max_features` is halved and the registration retried, down to
    [`MIN_MAX_FEATURES`][geotrax.utils.registration.MIN_MAX_FEATURES]; the detector-free `loftr`
    ignores `max_features` and is not retried.

    Args:
        img_src: Source image, e.g. a video frame.
        img_dst: Destination image, e.g. an orthophoto.
        logger: Receives the warnings about ignored keys, capped feature counts and retries.
        **stabilizer_kwargs: Any [`Stabilizer`](https://rfonod.github.io/stabilo/) parameter,
            typically the whole `georef.matching` block. Applied on top of
            [`DEFAULT_STABILIZER_KWARGS`][geotrax.utils.registration.DEFAULT_STABILIZER_KWARGS],
            so a key geo-trax does not know is still passed to stabilo, with a warning in case it
            is a typo. The keys in
            [`FIXED_STABILIZER_KWARGS`][geotrax.utils.registration.FIXED_STABILIZER_KWARGS] are
            pinned, and `max_features` is capped at
            [`DL_MAX_FEATURES_CAP`][geotrax.utils.registration.DL_MAX_FEATURES_CAP] for the
            learning-based detectors.

    Returns:
        `(H, inliers_count, num_matches, (n_src_kpts, n_dst_kpts))` on success, where `H` is the 3x3
            homography, `inliers_count` the number of RANSAC inliers and `num_matches` the number of
            filtered matches passed to the robust fit. `(None, None, None, None)` when every
            attempt failed.

    Raises:
        ValueError: From stabilo, for an invalid parameter combination. In particular `gpu=True`
            needs `detector_name='orb'` and a CUDA-enabled OpenCV build, and has no CPU fallback;
            the learning-based detectors use `device='cuda'` instead.
        RuntimeError: From a learning-based detector that runs out of memory (also `MemoryError`);
            stabilo re-raises it instead of reporting a failed registration. Lower `downsample_ratio`
            or `max_features`, or switch `device`.

    Examples:
        Register a drone frame to an orthophoto with the shipped matching settings:

        ```python
        import cv2
        from geotrax.utils.config_utils import load_config
        from geotrax.utils.logging_utils import setup_logger
        from geotrax.utils.registration import estimate_homography

        logger = setup_logger("register", dry_run=True)
        matching = load_config("default", logger)["georef"]["matching"]
        frame, ortho = cv2.imread("frame.png"), cv2.imread("ortho.png")
        H, inliers, matches, _ = estimate_homography(frame, ortho, logger, **matching)
        ```
    """
    unknown = sorted(set(stabilizer_kwargs) - KNOWN_STABILIZER_KWARGS)
    if unknown:
        logger.warning(f"Passing unrecognized registration key(s) {unknown} to stabilo as is; check cfg -> georef -> "
                       f"matching for typos.")

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
