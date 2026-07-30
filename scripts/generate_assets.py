"""Regenerate the brand icons and the social preview image into web/.

    uv run --group assets python scripts/generate_assets.py

Outputs are committed, so this only needs running when the art changes.
Everything is rasterized by screenshotting Chromium at deviceScaleFactor 1,
which keeps the PNGs pixel-exact at the sizes crawlers and launchers ask for.

The icon is a bishop: Brendan's pick, and its silhouette survives 16px better
than the knight's. It is a CC0 bishop (public domain), deliberately not the
vendored cburnett bishop — that set is CC BY-SA, which is fine for rendering a
board but carries share-alike obligations you do not want attached to a logo.

Two things vary by machine, so expect a diff in the card (never the icons,
which are pure geometry) if you regenerate somewhere else: the mock asks for
`system-ui`, which resolves to a different face per OS, and a Chromium version
bump can change rasterization. Neither is worth pinning a font file for; just
don't be surprised, and don't commit a card you only meant to look at.
"""

import re
import struct
import subprocess
from base64 import b64encode
from pathlib import Path
from shutil import which

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def optimize(path: Path, keep_rgba: bool = False) -> None:
    """Recompress a PNG in place, losslessly.

    Chromium's encoder is tuned for speed, so what it hands back is 13-45%
    larger than the same pixels need to be — on icons every browser fetches,
    and a card unfurlers fetch. Nearly all of that is won at `-o2`; `-o7` is
    here because it costs seconds on a script that runs when the art changes,
    and buys bytes on files every visitor pays for. `-strip all` is a no-op
    against today's Chromium, which writes no ancillary chunks; it is there so
    a version that starts writing a timestamp can't turn every regeneration
    into a diff.

    `keep_rgba` forbids reducing to a palette. The ICO's directory hardcodes
    32bpp for its members, and on these two files staying RGBA is smaller
    anyway — a palette plus the tRNS chunk costs more than it saves that small.
    """
    flags = ["-quiet", "-o7", "-strip", "all"] + (["-nc"] if keep_rgba else [])
    subprocess.run(["optipng", *flags, path], check=True)


def app_palette() -> str:
    """The app's own `:root` block, to render the card with.

    The card is an advertisement for the app, so it has to be showing the
    app's current colors rather than a copy taken whenever it was last drawn.
    """
    root = re.search(r":root\s*\{.*?\}", (WEB / "style.css").read_text(), re.S)
    if root is None:
        raise SystemExit("no :root block in web/style.css")
    return root.group(0)


# A light square with a dark piece on it. Rendered at 16px this beat the
# inverse and beat a colored tile: it is the highest-contrast pairing, and it
# is the only one that still reads as *chess* rather than as a generic glyph.
BG = "#f0d9b5"  # the board's light square
GLYPH = "#262421"

# CC0 bishop silhouette, drawn in a 297.08-unit square. Its ink occupies
# x 82.54..214.54, y 0..297.08 — measured with getBBox, and needed to center it.
BISHOP_PATH = (
    "M206.873,255.08h-3.41c2.214-3.337,8.32-14.536-0.712-25.6c-8.9-10.905-25.137-39.546-24.448-64.4"
    "h3.57c4.418,0,7.667-3.582,7.667-8v-1c0-4.418-3.249-8-7.667-8h-4.333v-4.285c13-8.971,20.511-23."
    "502,20.511-39.914c0-10.332-7.011-26.11-15.819-41.721l-18.553,18.595c-3.111,3.111-8.182,3.111-1"
    "1.294,0.001l-0.921-0.933c-3.111-3.11-3.106-8.202,0.005-11.313l21.676-21.674c-4.444-7.069-8.869"
    "-13.678-12.703-19.224c2.881-2.93,4.663-6.944,4.663-11.379C165.106,7.268,157.841,0,148.874,0c-8"
    ".967,0-16.234,7.268-16.234,16.233c0,4.434,1.781,8.448,4.662,11.379c-14.585,21.101-37.94,57.587"
    "-37.94,76.269c0,16.853,8.178,31.724,21.178,40.625v3.574h-4.667c-4.418,0-8.333,3.582-8.333,8v1c"
    "0,4.418,3.915,8,8.333,8h3.571c0.689,24.855-15.547,53.495-24.448,64.4c-9.031,11.064-2.926,22.26"
    "3-0.712,25.6h-3.411c-4.418,0-8.333,3.582-8.333,8v9c0,4.078,3,7.438,7,7.931v17.069h118v-17.069c"
    "4-0.493,7-3.853,7-7.931v-9C214.54,258.662,211.291,255.08,206.873,255.08z"
)
BISHOP_INK = {"x": 82.54, "y": 0.0, "w": 132.0, "h": 297.08}

SIDE = 512  # the SVGs' coordinate space; PNGs scale off it


def bishop_svg(glyph_height: float, corner_radius: float) -> str:
    """A square tile with the bishop centered on it.

    `glyph_height` is a fraction of the tile. Rounded corners suit a favicon,
    which nothing masks; a full-bleed square (radius 0) suits the icons whose
    platform rounds them itself, and those have to leave the glyph room to
    survive that crop.
    """
    scale = SIDE * glyph_height / BISHOP_INK["h"]
    tx = SIDE / 2 - scale * (BISHOP_INK["x"] + BISHOP_INK["w"] / 2)
    ty = SIDE / 2 - scale * (BISHOP_INK["y"] + BISHOP_INK["h"] / 2)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIDE} {SIDE}" '
        f'width="{SIDE}" height="{SIDE}">'
        f'<rect width="{SIDE}" height="{SIDE}" rx="{corner_radius:g}" fill="{BG}"/>'
        f'<path transform="translate({tx:.3f} {ty:.3f}) scale({scale:.5f})" '
        f'fill="{GLYPH}" d="{BISHOP_PATH}"/>'
        "</svg>"
    )


ROUNDED = bishop_svg(glyph_height=0.70, corner_radius=SIDE * 0.22)
# 0.55 keeps the glyph inside the maskable safe zone (the inner 80%).
MASKABLE = bishop_svg(glyph_height=0.55, corner_radius=0)
# iOS rounds the corners itself but crops nothing else, so this can run larger
# than a maskable icon — at 0.55 it looks lost on a home screen.
APPLE = bishop_svg(glyph_height=0.66, corner_radius=0)

ICO_SIZES = (16, 32)  # the PNGs favicon.ico wraps, so they stay RGBA

PNGS = [
    ("favicon-16x16.png", 16, ROUNDED),
    ("favicon-32x32.png", 32, ROUNDED),
    ("apple-touch-icon.png", 180, APPLE),
    ("android-chrome-192x192.png", 192, ROUNDED),
    ("android-chrome-512x512.png", 512, ROUNDED),
    ("android-chrome-192x192-maskable.png", 192, MASKABLE),
    ("android-chrome-512x512-maskable.png", 512, MASKABLE),
]


def write_ico(path: Path, members: list[tuple[int, bytes]]) -> None:
    """An ICO wrapping already-encoded PNGs, which every browser still probing
    a bare /favicon.ico has understood since Vista. Saves rasterizing twice."""
    header = struct.pack("<HHH", 0, 1, len(members))  # reserved, type=icon, count
    offset = len(header) + 16 * len(members)
    entries, blobs = b"", b""
    for size, png in members:
        # width, height, palette size, reserved, color planes, bpp, bytes, offset
        entries += struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
        blobs += png
    path.write_bytes(header + entries + blobs)


def main() -> None:
    if which("optipng") is None:
        raise SystemExit("optipng not on PATH — install it, or the assets ship oversized")

    written = [WEB / "favicon.svg"]
    (WEB / "favicon.svg").write_text(ROUNDED + "\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # deviceScaleFactor 1: a screenshot pixel is an output pixel.
        page = browser.new_page(device_scale_factor=1)

        for name, size, svg in PNGS:
            src = "data:image/svg+xml;base64," + b64encode(svg.encode()).decode()
            # <img> rather than inline SVG so the tile scales as one unit.
            page.set_viewport_size({"width": size, "height": size})
            page.set_content(
                "<style>html,body{margin:0}img{display:block;width:100vw;height:100vh}</style>"
                f'<img src="{src}">',
                wait_until="load",
            )
            # The rounded tile's corners must stay transparent.
            page.screenshot(path=WEB / name, omit_background=True)
            optimize(WEB / name, keep_rgba=size in ICO_SIZES)
            written.append(WEB / name)

        # Straight goto, so the mock's relative links (the cburnett CSS, the
        # favicon just written) resolve against its own directory.
        page.set_viewport_size({"width": 1200, "height": 630})
        page.goto((ROOT / "scripts" / "social-preview.html").as_uri())
        # Appended to <head>, so it outranks the mock's own :root on order.
        page.add_style_tag(content=app_palette())
        page.screenshot(path=WEB / "social-preview.png")
        optimize(WEB / "social-preview.png")
        written.append(WEB / "social-preview.png")

        browser.close()

    # Nothing links favicon.ico — the modern links above cover real browsers.
    # It exists because unfurlers and feed readers still probe the bare path.
    # Reading the files back means it wraps the optimized bytes, not the raw ones.
    write_ico(
        WEB / "favicon.ico",
        [(s, (WEB / f"favicon-{s}x{s}.png").read_bytes()) for s in ICO_SIZES],
    )
    written.append(WEB / "favicon.ico")

    for path in written:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
