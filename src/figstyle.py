"""Shared figure style for the M1 data figures (F3-F8).

Design rules (the dataviz method, adapted to print):
  * Two evaluators keep one colour each in every figure (original = blue, folding = orange);
    a colour never changes meaning between figures. Both are validated categorical slots 1 and 2
    (CVD dE 24.7, normal-vision dE 33.6 on a white surface, contrast >= 3:1).
  * Shape is the second channel (circle = original, square = folding), so grayscale print and
    colour-vision deficiency never rely on hue alone.
  * Text wears ink tokens, never the series colour. Hairline recessive grid on the value axis only.
    One y-axis per panel (no dual axes).
  * Figures are drawn at their final printed width so the type is a true 7.5-9 pt.
"""
from __future__ import annotations

import matplotlib as mpl

# ink / chrome tokens
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5df"
SURFACE = "#ffffff"

# entity colours (validated categorical slots 1-2) and their light tints
ORIG = "#2a78d6"
ORIG_LT = "#9ec5f4"
FOLD = "#eb6834"
FOLD_LT = "#f6bba4"
# status colour: reserved, always paired with a label
CRIT = "#d03b3b"
CRIT_LT = "#eba5a5"

# printed width of a 0.7\linewidth float on the draft's 460.9 pt text block
W_COL = 4.47


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
    })


def value_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def ringed(**kw) -> dict:
    """Marker kwargs: filled in the series colour with a 1.2 pt surface ring, so marks stay legible
    where they cross a line or overlap."""
    base = dict(markeredgecolor=SURFACE, markeredgewidth=1.2, markersize=6.5, zorder=4)
    base.update(kw)
    return base
