"""Content-addressed URLs for everything the frontend loads.

An asset that can't change under a URL can be cached forever, which is the only
way to stop a returning visitor spending a round trip per file asking whether it
changed. So every reference to one carries a digest of what it points at, and
the answer to "has this changed" is in the URL rather than in a request.

Computed here rather than by a build step, because the whole frontend is
build-free and a step you have to remember to run is a step that will be
forgotten — this cannot drift from what is on disk, since it reads the disk. The
tree is small enough (a few hundred KB) that holding it in memory is cheaper
than opening the files again per request.

HTML is deliberately left unversioned and uncacheable: it is the entry point, so
something has to be fetched to learn the digests, and that something is this.
"""

import hashlib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Asset:
    body: bytes
    media_type: str
    digest: str | None  # None for the entry points, which aren't versioned

    def cache_control(self, asked_for: str | None) -> str:
        if self.digest is None:
            return ENTRY_POINT
        return IMMUTABLE if asked_for == self.digest else UNVERSIONED


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
        assets["/" + path] = Asset(body, _media_type(path), digests.get(path))
    assets["/"] = assets["/index.html"]
    return assets
