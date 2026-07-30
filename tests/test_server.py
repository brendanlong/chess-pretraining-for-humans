import struct
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from trainer import auth, server

from .conftest import answer, next_trial


class Head(HTMLParser):
    """A served page's tags: metas by (attr, key), link hrefs, and the title."""

    @classmethod
    def of(cls, html: str) -> "Head":
        page = cls()
        page.feed(html)
        return page

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[tuple[str, str], str] = {}
        self.links: set[str] = set()
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attr = {k: v for k, v in attrs if v is not None}
        if tag == "meta":
            for key in ("name", "property"):
                if key in attr:
                    self.meta[(key, attr[key])] = attr.get("content", "")
        elif tag == "link" and attr.get("href"):
            self.links.add(attr["href"])
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def png_size(response) -> tuple[int, int]:
    """A PNG's real dimensions: 8-byte signature, 4-byte chunk length, the
    literal "IHDR", then width and height."""
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", response.content[16:24])


def user_row(conn, client):
    user = auth.session_user(conn, client.cookies[auth.COOKIE_NAME])
    assert user is not None
    return user


def test_no_repeats_until_exhausted_then_flagged(client, db):
    seen = set()
    for _ in range(2):
        t = next_trial(client)
        assert t["repeat"] is False
        assert t["item_id"] not in seen
        seen.add(t["item_id"])
        result = answer(client, t)
        assert result["repeat"] is False
        assert "correct" in result and "best" in result  # feedback on every trial

    # bank exhausted: repeats are flagged and rating-inert
    t = next_trial(client)
    assert t["repeat"] is True
    assert t["items_remaining"] == 0
    rating_before = user_row(db, client)["rating"]
    result = answer(client, t)
    assert result["repeat"] is True
    assert "correct" in result  # feedback still shown
    assert user_row(db, client)["rating"] == rating_before  # but no rating movement


def test_first_exposure_accuracy_excludes_repeats(client):
    for _ in range(4):  # 2 fresh + 2 repeats
        answer(client, next_trial(client))
    stats = client.get("/api/stats").json()
    assert stats["attempts"] == 4
    assert stats["first_exposures"] == 2
    assert len(stats["rating_history"]) == 2


def test_legal_pages_are_served_and_reachable_before_signing_up(client):
    """A guest records responses without ever opening the signup form, so the
    first page it lands on has to link the terms and the policy itself."""
    index = client.get("/")
    assert index.status_code == 200
    # The links in the signup form and the drawer don't count: both sit behind
    # a button the guest has no reason to press. The footer is the one that is
    # on screen next to the board.
    footer = index.text.split('id="page-footer"')[1].split("</footer>")[0]
    for page in ("terms.html", "privacy.html"):
        assert f'href="{page}"' in footer
        served = client.get(f"/{page}")
        assert served.status_code == 200
        assert served.headers["content-type"].startswith("text/html")


# Every tag a share needs to render as a card, plus the icon set. Hand-copied
# <head>s with no templating between them is how these drift apart, so the
# pages are globbed rather than listed: a fourth page is the failure mode, and
# a list would let one in unchecked.
SOCIAL_META = [
    ("name", "description"),
    ("name", "twitter:card"),
    ("property", "og:title"),
    ("property", "og:description"),
    ("property", "og:type"),
    ("property", "og:url"),
    ("property", "og:site_name"),
    ("property", "og:image"),
    ("property", "og:image:width"),
    ("property", "og:image:height"),
    ("property", "og:image:alt"),
]
ICON_LINKS = [
    "favicon.svg",
    "favicon-32x32.png",
    "favicon-16x16.png",
    "apple-touch-icon.png",
    "site.webmanifest",
]
PROD = "https://chess-pretraining.brendanlong.com"
WEB_PAGES = sorted(p.name for p in server.WEB_DIR.glob("*.html"))


def test_the_page_glob_still_finds_the_pages():
    """The tag checks below parametrize over this; an empty glob would pass
    every one of them by running none."""
    assert set(WEB_PAGES) >= {"index.html", "terms.html", "privacy.html"}


@pytest.mark.parametrize("name", WEB_PAGES)
def test_every_page_carries_the_social_and_icon_tags(client, name):
    path = "/" if name == "index.html" else f"/{name}"
    page = Head.of(client.get(path).text)

    for attr, key in SOCIAL_META:
        assert page.meta.get((attr, key), "").strip(), f"{name} is missing {key}"
    assert page.meta[("name", "twitter:card")] == "summary_large_image"
    # Crawlers resolve neither a relative og:image nor a relative og:url.
    assert page.meta[("property", "og:image")].startswith(PROD + "/")
    # Describing the page it is on, not whichever page it was copied from.
    assert page.meta[("property", "og:url")] == PROD + path
    assert page.meta[("property", "og:title")] == page.title
    assert page.meta[("property", "og:description")] == page.meta[("name", "description")]

    for href in ICON_LINKS:
        assert href in page.links, f"{name} is missing {href}"


@pytest.mark.parametrize("name", WEB_PAGES)
def test_every_page_declares_the_real_og_image_size(client, name):
    """A card whose declared dimensions don't match reserves the wrong box —
    and the image is regenerated by a script, so the numbers can fall behind."""
    path = "/" if name == "index.html" else f"/{name}"
    page = Head.of(client.get(path).text)
    served = client.get("/" + page.meta[("property", "og:image")].split("/")[-1])
    assert served.status_code == 200

    assert png_size(served) == (1200, 630)  # what every card renderer wants
    assert (
        page.meta[("property", "og:image:width")],
        page.meta[("property", "og:image:height")],
    ) == ("1200", "630")


def test_every_referenced_icon_is_served_at_its_declared_size(client):
    """The icons are loose files under web/; a rename would 404 in silence,
    and a resize would go on being advertised at the old size."""
    page = Head.of(client.get("/").text)
    manifest = client.get("/site.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["name"] == "Chess Pretraining"

    # Nothing links this one; unfurlers and feed readers probe the bare path.
    assert client.get("/favicon.ico").status_code == 200

    icons = manifest.json()["icons"]
    assert [i for i in icons if i.get("purpose") == "maskable"], "no maskable icon"
    # An installable icon has to be at least 192px square and unmasked.
    assert [i for i in icons if i["sizes"] == "192x192" and "purpose" not in i]

    for href in {*page.links, *(i["src"] for i in icons)}:
        if href.endswith((".png", ".svg", ".webmanifest", ".ico")):
            assert client.get("/" + href.lstrip("/")).status_code == 200, href
    for icon in icons:
        served = client.get(icon["src"])
        if icon["sizes"] != "any":  # the svg, which has no pixel size
            assert png_size(served) == tuple(int(n) for n in icon["sizes"].split("x"))


def test_healthz_is_free_and_anonymous(client, db):
    """The platform probes it every few seconds forever."""
    before = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}
    # No identity: a probe that minted a guest would fill the table with rows
    # the sweep then has to clear, and would hand the prober a session cookie.
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == before
    assert auth.COOKIE_NAME not in r.cookies


def test_separate_browsers_get_separate_identities(db):
    with TestClient(server.app) as a, TestClient(server.app) as b:
        answer(a, next_trial(a))
        assert a.get("/api/stats").json()["attempts"] == 1
        assert b.get("/api/stats").json()["attempts"] == 0
        assert next_trial(b)["repeat"] is False  # b's bank is untouched
