# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Color palettes for video visualization (VizColors) and trajectory plotting (PlotColors)."""

import random


class VizColors:
    """Color palette for video visualization."""
    def __init__(self) -> None:
        """Initialise the fixed RGB colour palette."""
        # Colours are indexed by class id, so the first four are the vehicle-class
        # colours (0=car blue, 1=bus red, 2=truck orange, 3=motorcycle green) and are
        # kept stable across releases. The palette is ordered most-distinct-first: the
        # first ten are saturated, well-separated hues; the last ten are lighter tints
        # of them, so colours become closer toward the end of the list.
        hexs = ('1F77B4', 'D62728', 'FF7F0E', '006400', '9467BD', '8C564B',
                '17BECF', 'E377C2', 'BCBD22', '7F7F7F', 'AEC7E8', 'FF9896',
                'FFBB78', '98DF8A', 'C5B0D5', 'C49C94', '9EDAE5', 'F7B6D2',
                'DBDB8D', 'C7C7C7')
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
    def __init__(self, colors: list | None = None) -> None:
        """Initialise with an optional list of hex colour strings."""
        self.colors = colors if colors else []

    def set_colors(self, colors: list) -> None:
        """Replace the current colour list."""
        self.colors = colors

    def get_color(self, index: int) -> str:
        """Return the hex colour for *index*, deriving a stable one beyond the configured list.

        plot.py calls this once per trajectory, so an index past the palette must return the same
        colour every time or a single source is drawn in many colours and its legend swatch, bound
        to the first line only, matches none of them. Seeding on the index makes the overflow
        colour a function of the index alone: identical across calls and across runs, and drawn
        from a private generator so the global random state stays untouched.
        """
        if index < len(self.colors):
            return self.colors[index]
        return f"#{random.Random(index).randrange(0x1000000):06x}"
