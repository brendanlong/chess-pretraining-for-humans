"""Links to our own pages have to navigate in place.

An installed PWA hands a new browsing context to an in-app browser, so a
`target="_blank"` on terms.html or privacy.html drops the user out of the app
to read them, with no way back but the OS. Off-site links still get it: leaving
for GitHub or Lichess is the point there.
"""

from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).parent.parent / "web"
PAGES = sorted(WEB.glob("*.html"))


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.links.append(dict(attrs))


def is_ours(href: str) -> bool:
    """Relative hrefs, and nothing else — we serve from a single origin."""
    return "://" not in href and not href.startswith(("mailto:", "#"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_own_links_stay_in_the_app(page: Path) -> None:
    collector = LinkCollector()
    collector.feed(page.read_text())
    escaping = [
        link["href"]
        for link in collector.links
        if (href := link.get("href")) and is_ours(href) and link.get("target") is not None
    ]
    assert not escaping, (
        f"{page.name} opens {escaping} in a new browsing context — "
        "an installed PWA sends that to an in-app browser"
    )


def test_there_are_pages_to_check() -> None:
    """So the parametrized test above can't pass by finding nothing."""
    assert PAGES
