# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Color palettes for video visualization (VizColors) and trajectory plotting (PlotColors)."""

import random
from typing import Optional


class VizColors:
    """Color palette for video visualization."""
    def __init__(self) -> None:
        """Initialise the fixed RGB colour palette."""
        hexs = ('4E79A7', 'F28E2B', 'E15759', '76B7B2', '59A14F', 'EDC948',
                'B07AA1', 'FF9DA7', '9C755F', 'BAB0AC', 'A0CBE8', 'FFBE7D',
                'FF9D9A', '86BCB6', '8CD17D', 'F1CE63', 'D4A6C8', 'FABFD2',
                'D7B5A6', 'C7C7C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.txt_color = (255, 255, 255)

    def __call__(self, i: int, bgr: bool = False) -> tuple:
        """Return palette colour at index i; swap to BGR channel order if bgr=True."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """Convert a '#RRGGBB' hex string to an (R, G, B) integer tuple."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class PlotColors:
    """Color palette for trajectory plotting."""
    def __init__(self, colors: Optional[list] = None) -> None:
        """Initialise with an optional list of hex colour strings."""
        self.colors = colors if colors else []

    def set_colors(self, colors: list) -> None:
        """Replace the current colour list."""
        self.colors = colors

    def get_color(self, index: int) -> str:
        if index < len(self.colors):
            return self.colors[index]
        else:
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))
