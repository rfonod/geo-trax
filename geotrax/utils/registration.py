#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Robert Fonod (robert.fonod@ieee.org) and Haechan Cho (gkqkemwh@kaist.ac.kr)

"""
registration.py - Image registration via Stabilo.

Estimates a homography between two images using the SIFT -> RootSIFT -> brute-force
knn matching -> Lowe ratio test -> MAGSAC++ pipeline. The computation is delegated to
the `stabilo` Stabilizer configured to reproduce that hand-rolled routine bit-for-bit
(see stabilo's `test_exact_reproduction_of_reference_routine`). This is the single
implementation shared by the georeferencing stage and the analysis tools.
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

    Uses a Stabilo Stabilizer configured to mirror the reference SIFT/RootSIFT/MAGSAC++
    routine exactly: the destination is set as the reference frame and the source as the
    current frame, so the resulting cur->ref transform maps src -> dst with the RANSAC
    reprojection threshold evaluated in destination coordinates.

    The defaults reproduce that routine bit-for-bit. `detector_name`, `matcher_name`,
    `filter_type` and `sift_enable_precise_upscale` are exposed for tuning; the geometry of
    the reproduction (projective transform, current-frame query, no masking/downsampling,
    1.0 reference multiplier) is fixed internally.

    If detection or matching fails, `max_features` is halved and retried (down to >10000),
    mirroring the original routine's fallback.

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
            num_matches = len(stabilizer.cur_inliers)  # total good matches (no getter yet; see stabilo TODO)
            return homography, inliers_count, num_matches, (n_src_kpts, n_dst_kpts)

        max_features_to_try //= 2
        logger.warning(
            f"SIFT detection or matching failed with {max_features_to_try * 2} max_features. "
            f"Trying with {max_features_to_try} max_features."
        )

    logger.error("SIFT detection failed with all attempted feature counts.")
    return None, None, None, None
