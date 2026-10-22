# `geotrax.utils.registration`

Homography estimation between two images through a [Stabilo](https://rfonod.github.io/stabilo/) `Stabilizer`: the single implementation behind georeferencing (frame to master frame, master frame to orthophoto) and the analysis tools. Feature detection, matching and the robust fit are delegated to Stabilo, so every detector it supports is available here; the [Feature detectors](../../guides/feature-detectors.md) guide compares them.

## Registration

::: geotrax.utils.registration.estimate_homography

## Settings

::: geotrax.utils.registration.DEFAULT_STABILIZER_KWARGS

::: geotrax.utils.registration.FIXED_STABILIZER_KWARGS

::: geotrax.utils.registration.KNOWN_STABILIZER_KWARGS

::: geotrax.utils.registration.MIN_MAX_FEATURES

::: geotrax.utils.registration.DL_MAX_FEATURES_CAP

## Detectors and devices

::: geotrax.utils.registration.DETECTOR_CHOICES

::: geotrax.utils.registration.DL_DETECTORS

::: geotrax.utils.registration.DEVICE_CHOICES
