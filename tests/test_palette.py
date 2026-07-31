"""The palette carries meaning, so it gets held to a number rather than an eye.

Colours are read out of the stylesheet, pushed through the Machado (2009)
dichromacy matrices and the channel gains a warm night-mode filter applies,
and the two that have to stay apart are checked for a minimum CIELAB
separation. See DESIGN.md for which pair is held to what and why they differ.
"""

import re
from pathlib import Path

import pytest

STYLE = Path(__file__).parent.parent / "web" / "style.css"

# Machado et al. 2009, severity 1.0, applied in linear RGB.
DICHROMACY = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
# Linear-light gains roughly matching f.lux at 2700K and at its 1900K extreme.
WARMTH = {"day": (1.0, 1.0, 1.0), "2700K": (1.0, 0.78, 0.52), "1900K": (1.0, 0.60, 0.26)}

# The board squares an arrow is drawn over, and the opacity it is drawn at.
BOARD = ("#f0d9b5", "#c0ae91")
ARROW_ALPHA = 0.8


def to_linear(c):
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_rgb(hex_color):
    return [to_linear(int(hex_color[i : i + 2], 16)) for i in (1, 3, 5)]


def over(fg, bg, alpha):
    """What the eye receives when a translucent arrow sits on a board square."""
    pairs = zip(linear_rgb(fg), linear_rgb(bg), strict=True)
    return [alpha * f + (1 - alpha) * b for f, b in pairs]


def seen_as(linear, vision, warmth):
    lit = [c * g for c, g in zip(linear, WARMTH[warmth], strict=True)]
    if vision == "normal":
        return lit
    return [sum(m * c for m, c in zip(row, lit, strict=True)) for row in DICHROMACY[vision]]


def lab(linear):
    x, y, z = (
        sum(m * c for m, c in zip(row, linear, strict=True)) / w
        for row, w in zip(
            ((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505)),
            (0.95047, 1.0, 1.08883),
            strict=True,
        )
    )
    f = [v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116 for v in (x, y, z)]
    return (116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2]))


def separation(a, b, vision, warmth):
    la, lb = (lab(seen_as(c, vision, warmth)) for c in (a, b))
    return sum((x - y) ** 2 for x, y in zip(la, lb, strict=True)) ** 0.5


@pytest.fixture(scope="module")
def palette():
    root = re.search(r":root\s*\{(.*?)\}", STYLE.read_text(), re.S)
    assert root, "no :root block in style.css"
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", root.group(1)))


ALL_VISION = ("normal", *DICHROMACY)


def worst(palette, first, second, on, alpha, visions=ALL_VISION, warmths=WARMTH):
    return min(
        separation(over(palette[first], bg, alpha), over(palette[second], bg, alpha), v, w)
        for bg in on
        for v in visions
        for w in warmths
    )


# A number ties each arrow to its button as well, but colour is what says it at
# a glance and a disc a fifth of a square in radius is not a substitute, so the
# candidate pair still has to clear every combination, warm filter stacked on
# dichromacy included.
def test_candidate_arrows_survive_everything(palette):
    assert worst(palette, "--arrow-1", "--arrow-2", BOARD, ARROW_ALPHA) > 40


def test_candidate_buttons_survive_everything(palette):
    assert worst(palette, "--accent-1", "--accent-2", ("#211f1c",), 1.0) > 40


# The reveal says which move won in text too, so its pair is held to
# dichromacy alone. Under deuteranopia plus a strong warm filter it drops to
# single digits, and no green/red pair does better on a board this light.
def test_reveal_pair_survives_dichromacy(palette):
    assert worst(palette, "--arrow-good", "--arrow-bad", BOARD, ARROW_ALPHA, warmths=["day"]) > 20
    assert worst(palette, "--good", "--bad", ("#211f1c",), 1.0, warmths=["day"]) > 35


# The pairs this rules out: blue/purple and green/red both collapse to one hue.
def test_the_pairs_this_replaced_would_fail(palette):
    old = {"a": "#1a56c4", "b": "#8a2be2", "good": "#15781b", "bad": "#b02323"}
    assert worst(old, "a", "b", BOARD, ARROW_ALPHA, warmths=["day"]) < 10
    assert worst(old, "good", "bad", BOARD, ARROW_ALPHA, warmths=["day"]) < 10
