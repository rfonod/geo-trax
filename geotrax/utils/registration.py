#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Robert Fonod (robert.fonod@ieee.org)

"""
registration.py - Image registration via Stabilo.

Estimates a homography between two images by delegating feature detection, matching and
robust model fitting to a `stabilo` Stabilizer. The detector, matcher, filter and RANSAC
parameters are configurable. This is the single implementation shared by the
georeferencing stage and the analysis tools.
"""

import logging

import cv2
import numpy as np
from stabilo import Stabilizer


def estimate_homography(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    logger: logging.Logger,
    *,
    detector_name: str = 'rsift',
    matcher_name: str = 'bf',
    filter_type: str = 'ratio',
    sift_enable_precise_upscale: bool = True,
    max_features: int = 250000,
    filter_ratio: float = 0.55,
    ransac_method: int = cv2.USAC_MAGSAC,
    ransac_epipolar_threshold: float = 3.0,
    ransac_max_iter: int = 10000,
    ransac_confidence: float = 0.999999,
    rsift_eps: float = 1e-8,
) -> tuple:
    """
    Estimate the homography H mapping source -> destination image coordinates.

    The destination is set as the Stabilizer's reference frame and the source as the
    current frame, so the resulting cur->ref transform maps src -> dst with the RANSAC
    reprojection threshold evaluated in destination coordinates.

    The detector, matcher, filter and RANSAC parameters are configurable; the geometry of
    the registration (projective transform, current-frame query, no masking/downsampling,
    1.0 reference multiplier) is fixed internally.

    If detection or matching fails, `max_features` is halved and retried (down to >10000).

    Returns:
        (H, inliers_count, num_matches, (n_src_kpts, n_dst_kpts)) on success, or
        (None, None, None, None) on failure. `inliers_count` is the number of RANSAC inliers
        and `num_matches` is the number of good matches fed to findHomography (i.e. the
        'inliers_count out of num_matches matches' figures).
    """
    max_features_to_try = max_features
    while max_features_to_try > 10000:
        stabilizer = Stabilizer(
            detector_name=detector_name,
            matcher_name=matcher_name,
            filter_type=filter_type,
            transformation_type='projective',
            clahe=False,
            mask_use=False,
            downsample_ratio=1.0,
            ref_multiplier=1.0,
            max_features=max_features_to_try,
            filter_ratio=filter_ratio,
            rsift_eps=rsift_eps,
            sift_enable_precise_upscale=sift_enable_precise_upscale,
            match_query_frame='current',
            ransac_method=ransac_method,
            ransac_confidence=ransac_confidence,
            ransac_epipolar_threshold=ransac_epipolar_threshold,
            ransac_max_iter=ransac_max_iter,
        )
        stabilizer.set_ref_frame(img_dst)
        stabilizer.stabilize(img_src)
        homography = stabilizer.get_cur_trans_matrix()

        if homography is not None:
            n_dst_kpts, n_src_kpts = stabilizer.get_cur_num_keypoints()  # (ref=dst, cur=src)
            inliers_count = stabilizer.get_cur_inliers_count()
            num_matches = stabilizer.get_cur_num_matches()
            return homography, inliers_count, num_matches, (n_src_kpts, n_dst_kpts)

        max_features_to_try //= 2
        logger.warning(
            f"Feature detection or matching failed with {max_features_to_try * 2} max_features. "
            f"Trying with {max_features_to_try} max_features."
        )

    logger.error("Feature detection failed with all attempted feature counts.")
    return None, None, None, None
