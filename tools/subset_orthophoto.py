#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Haechan Cho (gkqkemwh@kaist.ac.kr)

"""
subset_orthophoto.py - Orthophoto Subset Extraction Tool

This script extracts subsets from large orthophotos around specified geographic locations.
It crops square regions from the orthophoto centered on each location, downscales them,
and saves the results as PNG files with corresponding metadata.

The tool reads geographic coordinates from a location dictionary, converts them to pixel
coordinates using the orthophoto's geospatial metadata, and performs efficient tiled
cropping from GeoTIFF files.

Usage:
  python tools/subset_orthophoto.py --orthophoto-filepath <path> \\
         --ortho-cutout-folder <path> --location-dict-filepath <path> [options]

Arguments:
  --orthophoto-filepath <path>    : Path to the large orthophoto GeoTIFF file to be subsetted.
  --ortho-cutout-folder <path>    : Path to the output folder for cropped orthophotos and metadata.
  --location-dict-filepath <path> : Path to JSON file mapping location names to geographic coordinates.

Options:
  -h, --help                      : Show this help message and exit.
  --crop-size <int>               : Square crop size in pixels (default: 15000).
  --scale-factor <float>          : Downscaling factor for output images (default: 0.533).

Examples:
1. Basic orthophoto subsetting:
   python tools/subset_orthophoto.py --orthophoto-filepath ortho.tif \\
          --ortho-cutout-folder output/ --location-dict-filepath locations.json

2. Custom crop size and scale factor:
   python tools/subset_orthophoto.py --orthophoto-filepath ortho.tif \\
          --ortho-cutout-folder output/ --location-dict-filepath locations.json \\
          --crop-size 10000 --scale-factor 0.5

Input:
- GeoTIFF orthophoto file with geospatial metadata (ModelTiepointTag, ModelPixelScaleTag)
- JSON location dictionary: {"location_name": [latitude, longitude], ...}

Output:
- PNG files for each location: {location_name}.png
- Text files with pixel coordinates: {location_name}_center.txt
- Orthophoto parameters file: ortho_parameters.txt (lng_0, lat_0, lng_scale, lat_scale)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tifffile import TiffFile


def process_ortho(orthophoto_filepath, ortho_cutout_folder, location_dict_filepath, crop_size, scale_factor):
    """
    Subset the orthophoto for each location in location_path and save the down-scaled one.
    """
    with open(location_dict_filepath, 'r') as f:
        location_dict = json.load(f)

    with TiffFile(orthophoto_filepath) as tif:
        tif_tags = {tag.name: tag.value for tag in tif.pages[0].tags.values()}

    lng_0, lat_0 = tif_tags['ModelTiepointTag'][3:5][0], tif_tags['ModelTiepointTag'][3:5][1]
    lng_scale, lat_scale = tif_tags['ModelPixelScaleTag'][0], tif_tags['ModelPixelScaleTag'][1]

    ortho_cutout_folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(ortho_cutout_folder / 'ortho_parameters.txt', np.array([lng_0, lat_0, lng_scale, -lat_scale]))

    for location, coords in location_dict.items():
        lat, lng = coords
        p_x, p_y = int((lng - lng_0) / lng_scale), -int((lat - lat_0) / lat_scale)

        np.savetxt(ortho_cutout_folder / f'{location}_center.txt', np.array([p_x, p_y]))

        with TiffFile(orthophoto_filepath) as tif:
            page = tif.pages[0]
            cropped = get_tiled_crop(page, p_y - int(crop_size / 2), p_x - int(crop_size / 2), crop_size, crop_size)

        cropped_resized = cv2.resize(cropped, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        png_output_filename = ortho_cutout_folder / f'{location}.png'
        cv2.imwrite(str(png_output_filename), cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR))

        print(f"Processed and saved orthophoto for intersection: {location}")


def get_tiled_crop(page, i0, j0, h, w):
    """
    Subset the tiled geotiff file with initial location (i0, j0) and dimension (h, w).
    """
    if not page.is_tiled:
        raise ValueError("Input page must be tiled")

    im_width = page.imagewidth
    im_height = page.imagelength
    im_pyramid = page.imagedepth
    im_dim = page.samplesperpixel

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area [({i0}, {i1}), ({j0}, {j1})] is out of image bounds ({im_height}, {im_width})")

    tile_width, tile_height = page.tilewidth, page.tilelength

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty((im_pyramid,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    im_dim), dtype=page.dtype)

    fh = page.parent.filehandle

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables=page.jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    return out[0, im_i0: im_i0 + h, im_j0: im_j0 + w, :]


def parse_opt():
    parser = argparse.ArgumentParser(description="Subset large orthophotos around specified geographic locations.")
    parser.add_argument('--orthophoto-filepath', type=Path, help='Filepath to the orthophoto file to be subsetted.')
    parser.add_argument('--ortho-cutout-folder', type=Path, help='Path to the folder to save the cut orthophotos and meta files.')
    parser.add_argument('--location-dict-filepath', type=Path, help='Filepath to the location dictionary that maps location name to its geographic coordinates.')
    parser.add_argument('--crop-size', type=int, default=15000, help='Square crop size of the orthophoto (in pixels)')
    parser.add_argument('--scale-factor', default=8/15, type=float, help='Factor by which to downscale the cropped orthophoto before saving.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    process_ortho(**vars(opt))
