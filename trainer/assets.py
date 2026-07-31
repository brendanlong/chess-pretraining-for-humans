"""Content-addressed, pre-compressed URLs for everything the frontend loads.

An asset that can't change under a URL can be cached forever, which is the only
way to stop a returning visitor spending a round trip per file asking whether it
changed. So every reference to one carries a digest of what it points at, and
the answer to "has this changed" is in the URL rather than in a request.

Both the digests and the compressed copies are computed here, at startup,
against whatever tree the server was pointed at — the sources in a dev checkout,
`scripts/build-web.mjs`'s bundled output in the image. The build step
deliberately does neither: it only makes files smaller and fewer, so there is
one implementation of what a URL means and what may be cached, and it reads the
disk rather than trusting a manifest that could disagree with it. Compressing
the whole tree at brotli's maximum costs about a tenth of a second once per
boot, which is not worth a second code path to avoid.

The tree is small enough (a few hundred KB) that holding it, and its compressed
variants, in memory is cheaper than opening the files again per request.

HTML is deliberately left unversioned and uncacheable: it is the entry point, so
something has to be fetched to learn the digests, and that something is this.
"""

import gzip
import hashlib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

import brotli

# A year, which is the convention for "forever"; `immutable` additionally stops
# a reload from revalidating it.
IMMUTABLE = "public, max-age=31536000, immutable"
# Reached without a digest — a bookmark, a crawler, a hand-typed URL — so it has
# no way to be told it changed. Cacheable, but not for long enough to strand
# anyone on a stale copy.
UNVERSIONED = "public, max-age=300"
# Carries the digests, so it can never be the stale copy.
ENTRY_POINT = "no-cache"

VERSION_PARAM = "v"
_TEXT = {".html", ".css", ".js", ".webmanifest", ".svg", ".json"}
# Under this, the framing and the extra header cost about what the compression
# saves. The files it excludes are icons and the manifest.
_MIN_COMPRESS = 512
# Tried in this order, so the server's preference wins rather than the client's:
# every browser that offers brotli also offers gzip, and would otherwise decide
# by a quality value it set arbitrarily.
_ENCODINGS = ("br", "gzip")


@dataclass(frozen=True)
class Asset:
    body: bytes
    media_type: str
    digest: str | None  # None for the entry points, which aren't versioned
    encoded: dict[str, bytes]  # content-encoding -> body; empty for the incompressible

    def cache_control(self, asked_for: str | None) -> str:
        if self.digest is None:
            return ENTRY_POINT
        return IMMUTABLE if asked_for == self.digest else UNVERSIONED

    def negotiate(self, accept_encoding: str) -> tuple[bytes, str | None]:
        """The best copy this client will take, and what to call it."""
        accepted = _accepted(accept_encoding)
        for encoding in _ENCODINGS:
            if encoding in accepted and encoding in self.encoded:
                return self.encoded[encoding], encoding
        return self.body, None


def _accepted(header: str) -> set[str]:
    """Encoding names the client will take, dropping any it explicitly refused.

    `br;q=0` means "not this one", and is the whole reason to parse rather than
    substring-match the header: a client that ruled brotli out has to be handed
    something else, and answering with it anyway is a response it can't read.
    """
    accepted = set()
    for part in header.split(","):
        name, _, params = part.strip().partition(";")
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        if quality > 0:
            accepted.add(name.strip().lower())
    return accepted


def _compress(path: str, body: bytes) -> dict[str, bytes]:
    """Every encoding worth offering for one file, at maximum effort.

    Affordable only because it happens once at startup rather than per request,
    which is also what makes the maximum the right setting: the usual reason to
    compress lightly is that the CPU is on the request path, and here it isn't.
    """
    if not path.endswith(tuple(_TEXT)) or len(body) < _MIN_COMPRESS:
        return {}
    # mtime=0 so a gzip body is a function of its input alone — two boots of the
    # same tree should not disagree about any byte they serve.
    candidates = {
        "br": brotli.compress(body, quality=11),
        "gzip": gzip.compress(body, 9, mtime=0),
    }
    return {name: packed for name, packed in candidates.items() if len(packed) < len(body)}


def _digest(body: bytes) -> str:
    # Truncated because this is a cache key, not a signature: 48 bits is far
    # past coincidence for twenty files, and short URLs are easier to read.
    return hashlib.sha256(body).hexdigest()[:12]


def _media_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _reference(path: str) -> re.Pattern[str]:
    """Matches the path as it appears in a reference, however it's spelled.

    Anchored on the delimiters around it so that a name can't match inside a
    longer one, and so a bare word in prose is left alone.
    """
    return re.compile(rf"""(["'(])((?:\./|/)?{re.escape(path)})(["')])""")


def build(root: Path) -> dict[str, Asset]:
    """Every file under `root`, keyed by its URL path, references rewritten."""
    bodies = {
        str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    # Only the entry points go unversioned, which also means nothing versioned
    # can reference something unversioned — so a single pass in dependency order
    # terminates, with no cycle to worry about between the pages that link to
    # each other.
    versioned = sorted(p for p in bodies if not p.endswith(".html"))
    digests: dict[str, str] = {}

    def rewrite(body: bytes, known: dict[str, str]) -> bytes:
        text = body.decode()
        for path, digest in known.items():
            text = _reference(path).sub(rf"\1\2?{VERSION_PARAM}={digest}\3", text)
        return text.encode()

    remaining = list(versioned)
    while remaining:
        progressed = False
        for path in list(remaining):
            depends_on = [
                other
                for other in remaining
                if other != path
                and path.endswith(tuple(_TEXT))
                and _reference(other).search(bodies[path].decode(errors="ignore"))
            ]
            if depends_on:
                continue
            if path.endswith(tuple(_TEXT)):
                bodies[path] = rewrite(bodies[path], digests)
            digests[path] = _digest(bodies[path])
            remaining.remove(path)
            progressed = True
        if not progressed:  # only reachable if two assets reference each other
            raise RuntimeError(f"reference cycle among {remaining}")

    assets = {}
    for path, body in bodies.items():
        if path.endswith(".html"):
            body = rewrite(body, digests)
        assets["/" + path] = Asset(
            body, _media_type(path), digests.get(path), _compress(path, body)
        )
    assets["/"] = assets["/index.html"]
    return assets
