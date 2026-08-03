import ast
import contextlib
import gzip
import json
import re
import sqlite3
import struct
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import brotli
import pytest
from fastapi.testclient import TestClient

from trainer import assets, auth, rating, server, trials
from trainer.db import connect

from .conftest import FEN_TMPL, ITEM, add_item, answer, answer_body, next_trial


class Head(HTMLParser):
    """A served page's tags: metas by (attr, key), link and anchor hrefs, the
    scripts, and the title."""

    @classmethod
    def of(cls, html: str) -> "Head":
        page = cls()
        page.feed(html)
        return page

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[tuple[str, str], str] = {}
        self.links: set[str] = set()
        self.anchors: list[dict[str, str]] = []
        self.scripts: list[dict[str, str]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attr = {k: ("" if v is None else v) for k, v in attrs}
        if tag == "meta":
            for key in ("name", "property"):
                if key in attr:
                    self.meta[(key, attr[key])] = attr.get("content", "")
        elif tag == "link" and attr.get("href"):
            # Without the digest: what these assertions are about is which files
            # a page pulls in, not the version it pulls them in at.
            self.links.add(attr["href"].split("?")[0])
        elif tag == "a" and "href" in attr:
            self.anchors.append(attr)
        elif tag == "script":
            self.scripts.append(attr)
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


def unanswered(conn, client) -> int:
    """Servable items this client has never answered — what selection has left
    to draw from. The app itself never asks: selection filters rather than
    counts, so a test that cares what a trial consumed asks the bank straight
    out. Per user, because "nobody has answered it" is a different claim and
    not the one selection acts on."""
    return conn.execute(
        "SELECT COUNT(*) FROM items WHERE learnable = 1"
        " AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)",
        (user_row(conn, client)["id"],),
    ).fetchone()[0]


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
    assert unanswered(db, client) == 0
    t = next_trial(client)
    assert t["repeat"] is True
    rating_before = user_row(db, client)["rating"]
    result = answer(client, t)
    assert result["repeat"] is True
    assert "correct" in result  # feedback still shown
    assert user_row(db, client)["rating"] == rating_before  # but no rating movement


# Everything about how hard the item is, or how that was measured. Each one is
# a hint about where to look, and `shallow_gap` is worse than a hint: its *sign*
# is which move a shallow search prefers, so anyone with an engine reads the
# answer key straight off it. Listed rather than checked one at a time so that
# a column added to a payload later has to be added here too.
NEVER_BEFORE_ANSWERING = ("shallow_gap", "gap_ladder", "gap_wp", "item_rating")
# The subset the reveal does say. Asserted too, so that deleting a field from
# the reveal can't quietly turn the guard above into a test of nothing.
SAID_BY_THE_REVEAL = ("item_rating",)


def test_what_the_reveal_says_about_difficulty_never_reaches_the_trial(client):
    """How hard the item was is worth saying once it can't help — but the
    measurement behind it stays in the bank, on either side of the answer."""
    trial = json.dumps(next_trial(client))
    for field in NEVER_BEFORE_ANSWERING:
        assert field not in trial, f"{field} leaked into the trial payload"

    revealed = answer(client, next_trial(client))
    for field in SAID_BY_THE_REVEAL:
        assert field in revealed, f"{field} is not in the reveal, so this guards nothing"
    # Serialized, like the trial above: a measurement would most naturally be
    # added per move, inside `best`/`distractor`, where a key check can't see it.
    revealed_text = json.dumps(revealed)
    for field in set(NEVER_BEFORE_ANSWERING) - set(SAID_BY_THE_REVEAL):
        assert field not in revealed_text, f"{field} is in the reveal and nothing reads it"
    assert revealed["item_rating"] == round(ITEM["rating"])


def test_first_exposure_accuracy_excludes_repeats(client):
    """Repeats are answerable from memory of the reveal, so they say nothing
    about skill and must not move the reported accuracy either way."""

    def answer_with(trial, uci):
        answer(client, trial, [m["uci"] for m in trial["moves"]].index(uci))

    for _ in range(2):  # the whole bank, answered correctly
        answer_with(next_trial(client), ITEM["best_uci"])
    assert client.get("/api/stats").json()["accuracy_window"] == [1, 1]

    for _ in range(2):  # now repeats, answered wrongly
        t = next_trial(client)
        assert t["repeat"] is True
        answer_with(t, ITEM["distractor_uci"])
    stats = client.get("/api/stats").json()
    assert stats["attempts"] == 4  # all four were recorded
    assert stats["accuracy_window"] == [1, 1]  # but only the two fresh ones counted


@pytest.mark.parametrize("item_count", [server.ACCURACY_WINDOW + 5])
def test_the_accuracy_window_is_the_newest_answers_oldest_first(client, item_count):
    """The client extends this window as it answers, so both ends of it are a
    contract: it has to arrive capped at the width the client keeps trimming
    to, and ordered so that appending is what adds the newest answer."""
    wrong_at = {3, item_count - 1}  # one that falls out of the window, one that can't
    for i in range(item_count):
        trial = next_trial(client)
        uci = ITEM["distractor_uci"] if i in wrong_at else ITEM["best_uci"]
        answer(client, trial, choice_index_of(trial, uci))

    window = client.get("/api/stats").json()["accuracy_window"]
    assert len(window) == server.ACCURACY_WINDOW  # the five oldest dropped out
    assert window[-1] == 0  # newest last: the miss just answered
    assert sum(window) == server.ACCURACY_WINDOW - 1  # and only that one: the early miss fell out


def last_response(db):
    return db.execute("SELECT * FROM responses ORDER BY id DESC LIMIT 1").fetchone()


def test_the_row_says_whether_the_staircase_owned_the_rating(client, db):
    """`calibrating` is the staircase's state as the answer was scored, so the
    fits over `responses` can hold those moves out instead of inferring them
    from the delta — the inference breaks where a bound clamps the move."""
    answer(client, next_trial(client))
    assert last_response(db)["calibrating"] == 1  # a fresh user is mid-staircase
    with db:
        db.execute("UPDATE users SET calib_step = 1")  # calibration over
    answer(client, next_trial(client))
    assert last_response(db)["calibrating"] == 0


def test_a_young_rating_moves_at_the_provisional_k(client, db):
    """A rating resting on a staircase's handful of answers has to move by
    more than the settled K, or a calibration exit luck got wrong takes
    hundreds of answers to walk back."""
    t = next_trial(client)
    answer(client, t, choice_index_of(t, ITEM["best_uci"]))  # mints the user
    with db:
        db.execute("UPDATE users SET calib_step = 1")  # calibration over, one answer old
    t = next_trial(client)
    r = answer(client, t, choice_index_of(t, ITEM["distractor_uci"]))
    # A miss at the settled K can cost at most K_USER; only k_factor's boost
    # reaches past it.
    assert r["rating_delta"] < -rating.K_USER


def test_a_clock_reading_that_cannot_be_one_is_recorded_as_none(client, db):
    """`response_ms` is client-supplied; garbage is kept as "not measured"
    rather than believed — or bounced, which would throw away a real answer
    over a timestamp."""
    for sent, kept in ((-5, None), (4_200, 4_200), (server.RESPONSE_MS_MAX + 1, None)):
        body = {**answer_body(next_trial(client)), "response_ms": sent}
        assert client.post("/api/answer", json=body).status_code == 200
        assert last_response(db)["response_ms"] == kept


# --- share links ----------------------------------------------------------
#
# A URL names an item, so it reaches one selection would not have offered. The
# payload is the same symmetric one every trial gets, so naming an item buys
# the position and the pair and never the answer; the answer counts like any
# other, and carries a mark saying nobody aimed this one.


def choice_index_of(trial, uci) -> int:
    """Which button that move is on this trial — the pair is shuffled per trial."""
    return [m["uci"] for m in trial["moves"]].index(uci)


def other_item(db, trial) -> int:
    """An item the trial on screen isn't, so a share of it is a fresh one."""
    return db.execute("SELECT id FROM items WHERE id != ?", (trial["item_id"],)).fetchone()[0]


def shared_trial(client, item_id):
    r = client.get(f"/api/next?item={item_id}")
    assert r.status_code == 200, r.text
    return r.json()


def test_a_share_link_serves_the_item_it_names(client, db):
    wanted = db.execute("SELECT id FROM items ORDER BY id DESC").fetchone()[0]
    # Ask enough times that ordinary selection landing on it would be a
    # coincidence rather than the thing being tested.
    for _ in range(5):
        trial = shared_trial(client, wanted)
        assert trial["item_id"] == wanted
        # Still a trial like any other: nothing about which move is better.
        assert not set(NEVER_BEFORE_ANSWERING) & set(trial)


def test_an_answer_from_a_share_link_counts_like_any_other_and_is_marked(client, db):
    """The mark is the whole of what sharing costs: the answer is a real first
    exposure, so it rates and it counts, and what the analysis needs is to be
    able to hold out the rows nobody aimed.

    Answered *wrongly*, because that is what tells the two designs apart: an
    answer held out of the rating and the accuracy leaves both exactly where
    they were, and a rating that only ever moves up would agree with either.
    """
    first = next_trial(client)
    answer(client, first, choice_index_of(first, ITEM["best_uci"]))
    before = user_row(db, client)["rating"]
    shared = shared_trial(client, other_item(db, first))

    result = answer(client, shared, choice_index_of(shared, ITEM["distractor_uci"]))
    assert result["correct"] is False
    assert "best" in result  # feedback, like any trial
    assert user_row(db, client)["rating"] < before  # and a rating that moved
    row = db.execute("SELECT * FROM responses ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["item_id"], row["shared"]) == (shared["item_id"], 1)
    # Counted where every other first exposure is, too.
    stats = client.get("/api/stats").json()
    assert stats["attempts"] == 2
    assert stats["accuracy_window"] == [1, 0]  # one right, one wrong — both counted
    assert unanswered(db, client) == 0  # and consumed, like any first exposure


def test_a_shared_answer_is_scored_by_elo_even_during_calibration(client, db):
    """The staircase steps by a fixed amount *because* selection guarantees the
    item was aimed at the user. On an item nobody aimed it would pay a quarter
    of the scale for a two-alternative guess — so during calibration a shared
    answer goes through Elo, which reads how hard the item actually was."""
    first = next_trial(client)
    answer(client, first, choice_index_of(first, ITEM["best_uci"]))
    calibrating = user_row(db, client)
    # Still on the staircase, one win in: the step shrank but didn't halve.
    assert calibrating["calib_step"] == rating.CALIB_START_STEP * rating.CALIB_WIN_DECAY
    # The staircase's own move, for comparison: the full step, whatever the item.
    assert calibrating["rating"] == rating.USER_START + rating.CALIB_START_STEP

    shared = shared_trial(client, other_item(db, first))
    before = calibrating["rating"]
    answer(client, shared, choice_index_of(shared, ITEM["best_uci"]))
    after = user_row(db, client)

    # Elo's move at the one-answer-old provisional K — still nothing like the
    # step the staircase would have paid for an item nobody aimed.
    assert 0 < after["rating"] - before <= rating.k_factor(1)
    assert after["rating"] - before < calibrating["calib_step"]
    # The staircase is where it was: a trial it didn't choose doesn't advance it.
    assert after["calib_step"] == calibrating["calib_step"]


def test_a_url_naming_an_item_you_have_answered_reopens_it_as_a_rerun(client, db):
    """The case that isn't about links at all: the tab reloads, and the URL it
    reloads names the trial whose answer is on the screen it came from. It
    opens that position rather than a stranger's — answerable, and worth
    nothing, which is what makes serving a remembered answer safe.

    Answered *wrongly* the second time, because that is what tells the two
    designs apart: an answer held out of the rating leaves it where it was, and
    an unrated right answer would agree with either."""
    first = next_trial(client)
    answer(client, first, choice_index_of(first, ITEM["best_uci"]))
    before = user_row(db, client)["rating"]

    again = shared_trial(client, first["item_id"])
    assert again["item_id"] == first["item_id"]
    assert again["repeat"] is True
    result = answer(client, again, choice_index_of(again, ITEM["distractor_uci"]))

    assert result["correct"] is False
    assert "best" in result  # feedback, like any trial
    assert result["repeat"] is True
    assert user_row(db, client)["rating"] == before  # and a rating that didn't move
    # Counted nowhere a fresh answer would be, either.
    stats = client.get("/api/stats").json()
    assert stats["accuracy_window"] == [1]  # the wrong rerun is not a first exposure
    assert unanswered(db, client) == 1  # and it consumed nothing: one still unseen
    # And still marked, which is how the analysis holds out what nobody aimed.
    row = db.execute("SELECT * FROM responses ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["item_id"], row["shared"]) == (first["item_id"], 1)


@pytest.mark.parametrize(
    "named",
    [
        "99999",  # an id from a bank this one isn't
        "0",
        "-1",
        "not-an-id",
        "12)",  # a link that went through a chat client
        "",
        "1e3",
        "١٢٣",  # digits `isdigit` accepts and `int` parses
        "1" * 20,  # past what SQLite can bind
        "9" * 5000,  # past what Python will parse
    ],
)
def test_a_url_the_bank_cannot_honour_still_opens_the_app(client, db, named):
    """A link outlives the bank it was made from, and travels through chat
    clients that hand it back with a bracket on the end. Every way of naming
    nothing lands on an ordinary trial, because what is in front of somebody
    who just followed a link should never be a stack trace or a validation
    error — and this endpoint is reachable by anyone, unmetered."""
    r = client.get(f"/api/next?item={named}")
    assert r.status_code == 200, r.text
    # Served something real, and not the thing that was asked for — which is
    # how the page knows to say the link couldn't be opened.
    assert r.json()["item_id"] != named
    assert db.execute("SELECT 1 FROM items WHERE id = ?", (r.json()["item_id"],)).fetchone()


@pytest.mark.parametrize("item_count", [1])
def test_a_link_to_an_unlearnable_item_is_not_honoured(client, db, item_count):
    """The one thing that is never served: an item whose answer the engine
    won't hold still has nothing to teach, and a link is not a way in."""
    add_item(db, FEN_TMPL.format("7P"), learnable=0)
    db.commit()
    unlearnable = db.execute("SELECT id FROM items WHERE learnable = 0").fetchone()[0]

    assert shared_trial(client, unlearnable)["item_id"] != unlearnable


def test_first_exposure_filter_is_answered_from_an_index_covering_item_id(db):
    """The filter asks, per response, whether an earlier one hit the same item —
    see `idx_responses_item` in `db.py` for what indexing it saves.

    Asserts the plan, not a duration, because a timing threshold flakes on CI.
    The losing plan is a SEARCH too — over a range rather than a row — so which
    index gets chosen is the whole assertion.
    """
    plan = db.execute("EXPLAIN QUERY PLAN " + server.RECENT_FIRST_EXPOSURES_SQL, (1,)).fetchall()
    inner = [row[-1] for row in plan if "p" in row[-1].split()]
    assert inner, f"no plan step for the inner query: {[r[-1] for r in plan]}"
    assert "idx_responses_item" in inner[0], inner[0]


def test_selection_walks_the_rating_index_instead_of_scanning_the_bank(db):
    """Serving a trial must not cost a pass over every item row — see
    `PICK_SQL` in server.py for what the walks replace.

    Asserts the plan, not a duration, because a timing threshold flakes on CI.
    The losing plan is `SCAN items`, and it is what any edit that stops the
    partial index applying — dropping the `learnable = 1` term, or ordering
    the walks by an expression — quietly reverts to. (A temp b-tree does
    appear: it sorts the merged walks, never more than two pools of rows.)
    """
    params = {"target": 1200, "user_id": 1, "k": 3}
    steps = [row[-1] for row in db.execute("EXPLAIN QUERY PLAN " + server.PICK_SQL, params)]
    walks = [s for s in steps if "idx_items_learnable_rating" in s]
    assert len(walks) == 2, steps  # one per direction
    assert all(s.startswith("SEARCH") for s in walks), steps
    # "SCAN items" on this SQLite; older ones say "SCAN TABLE items". Matching
    # both keeps the guard from going quiet under a different interpreter.
    assert not any(s.startswith("SCAN") and "items" in s for s in steps), steps


def test_the_index_walks_merge_to_the_pool_the_distance_ordering_chose(db):
    """The LIMITed walks are only an optimization if every OFFSET of `PICK_SQL`
    still lands inside the `SELECTION_POOL` nearest unseen items — the set a
    full `ORDER BY ABS(rating - target)` selects by construction."""
    from trainer.rating import difficulty_rating

    from .conftest import add_item

    # Distinct difficulties on both sides of the target, more than a pool per
    # side, with a user who has already seen some of the closest ones.
    for i in range(90):
        gap = 0.01 + 0.004 * i
        add_item(db, f"pool-test-{i}", shallow_gap=gap, rating=difficulty_rating(gap))
    with db:
        user = auth.create_guest(db, 850.0, 250.0)
        db.executemany(
            """INSERT INTO responses (user_id, item_id, choice_uci, correct)
               VALUES (?, ?, 'e2e4', 1)""",
            [(user["id"], item_id) for item_id in range(20, 40)],
        )
    target, uid = db.execute("SELECT AVG(rating), ? FROM items", (user["id"],)).fetchone()
    old = db.execute(
        """SELECT ABS(rating - ?) AS d FROM items
           WHERE learnable = 1 AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)
           ORDER BY d LIMIT ?""",
        (target, uid, server.SELECTION_POOL),
    ).fetchall()
    who = {"target": target, "user_id": uid}
    picked = [
        db.execute(server.PICK_SQL, {**who, "k": k}).fetchone()
        for k in range(server.SELECTION_POOL)
    ]
    # Distances rather than ids: equal-rated items tie, and any nearest-30 set
    # is as good as any other — what must match is how near the pool sits.
    assert sorted(abs(r["rating"] - target) for r in picked) == [r["d"] for r in old]


def test_a_pool_smaller_than_the_draw_is_redrawn_not_dropped(db, monkeypatch):
    """The first OFFSET is drawn as if the pool were full; a bank holding less
    must redraw over what exists rather than reporting exhaustion early."""
    monkeypatch.setattr(server, "conn", db)

    draws = []

    class MissFirst:
        """Always the highest draw, so the first OFFSET overshoots a bank of 2."""

        def randrange(self, n: int) -> int:
            draws.append(n)
            return n - 1

    monkeypatch.setattr(server, "rng", MissFirst())
    item, repeat = server.pick_item(850.0, None)
    assert item is not None and repeat is False
    # Missed at pool size, then redrawn over exactly the items that exist.
    assert draws == [server.SELECTION_POOL, 2]


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


def is_ours(href: str) -> bool:
    """A page of ours: a relative URL, or an absolute one at the production
    host — the same page either way, so both have to answer for themselves.
    `mailto:` and friends reach another app rather than a page.
    """
    parts = urlsplit(href)
    return parts.scheme in ("", "http", "https") and parts.netloc in ("", urlsplit(PROD).netloc)


@pytest.mark.parametrize("name", WEB_PAGES)
def test_our_own_pages_open_in_the_app(client, name):
    """These pages are the app, so they have to open in it.

    An installed app shows a new browsing context as an in-app browser sheet:
    our own terms would arrive wearing someone else's chrome, behind a Done
    button people reliably miss. Nothing here wants that — a `target` on one
    of our own links buys nothing back, since navigating in place loses
    nothing on a page with no state.

    Only off-site links are a judgment call, so only this direction is fixed;
    `index.html` says which way it went and why.
    """
    path = "/" if name == "index.html" else f"/{name}"
    escaping = [
        a["href"]
        for a in Head.of(client.get(path).text).anchors
        if is_ours(a["href"]) and "target" in a
    ]
    assert not escaping, f"{name} opens {escaping} in a new browsing context"


AUTHOR = "https://www.brendanlong.com/pages/about-me.html"
SOURCE = "https://github.com/brendanlong/chess-pretraining-for-humans"


def is_the_authors(href: str) -> bool:
    """His own site rather than this app's, which is a subdomain of it. The
    host is compared whole — a suffix alone would take `notbrendanlong.com`
    for his, and a bare substring would take any URL that merely mentions it.
    """
    host = urlsplit(href).netloc
    return (host == "brendanlong.com" or host.endswith(".brendanlong.com")) and not is_ours(href)


@pytest.mark.parametrize("name", WEB_PAGES)
def test_every_page_credits_the_author_at_one_address(client, name):
    """Every page says who made it and where the source is, and the author
    link is the same address everywhere: it is hand-copied into each footer
    and into the drawer, so the drift is a page pointing somewhere else on
    the same site — a home page or a stale path — which reads as working."""
    path = "/" if name == "index.html" else f"/{name}"
    hrefs = [a["href"] for a in Head.of(client.get(path).text).anchors]

    assert AUTHOR in hrefs, f"{name} doesn't credit the author"
    assert SOURCE in hrefs, f"{name} hides the source"
    strays = [h for h in hrefs if is_the_authors(h) and h != AUTHOR]
    assert not strays, f"{name} links the author at {strays}"


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
    # nothing reclaims, and would hand the prober a session cookie.
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == before
    assert auth.COOKIE_NAME not in r.cookies


def test_separate_browsers_get_separate_identities(db):
    with TestClient(server.app) as a, TestClient(server.app) as b:
        answer(a, next_trial(a))
        assert a.get("/api/stats").json()["attempts"] == 1
        assert b.get("/api/stats").json()["attempts"] == 0
        assert next_trial(b)["repeat"] is False  # b's bank is untouched


def test_an_answer_must_be_to_the_trial_that_was_served(client, db):
    """The /api/answer payload *is* the answer key — best move, both evals,
    both lines — and item ids are small sequential integers. Without this,
    reading the answer to the trial on your own screen before committing to it
    is one request, which is the one thing SPEC says nothing may do; and the
    whole bank can be dumped by counting.
    """
    served = next_trial(client)
    other = db.execute("SELECT id FROM items WHERE id != ?", (served["item_id"],)).fetchone()[0]

    r = client.post("/api/answer", json={"item_id": other, "choice_uci": "e2e4"})
    assert r.status_code == 409
    assert "best" not in r.json()  # nothing about the item comes back

    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    # ...and the trial actually in progress still answers fine.
    assert answer(client, served)["correct"] in (True, False)


def test_an_unserved_item_is_unanswerable_without_a_token(db):
    """The interesting caller isn't the one playing — it's a second client asking
    about the item id the first one is looking at."""
    with TestClient(server.app) as player, TestClient(server.app) as prober:
        trial = next_trial(player)
        r = prober.post(
            "/api/answer",
            json={"item_id": trial["item_id"], "choice_uci": trial["moves"][0]["uci"]},
        )
        assert r.status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_a_trial_issued_to_one_session_cannot_be_redeemed_by_another(client, db):
    """The token names its holder, not just its item. Without that, a throwaway
    client could fetch a token and the signed-in one could spend it — which is
    the pre-commit peek the binding exists to stop."""
    answer(client, next_trial(client))  # `client` now has an identity
    mine = next_trial(client)
    with TestClient(server.app) as other:
        assert other.post("/api/answer", json=answer_body(mine)).status_code == 409
    # And a *different* identity's token is no good to me either.
    with TestClient(server.app) as third:
        answer(third, next_trial(third))
        theirs = next_trial(third)
    assert client.post("/api/answer", json=answer_body(theirs)).status_code == 409


# The signed payload's fields, by position: item, user, served-as-repeat,
# served-as-shared, nonce, expiry — then the mac.
USER_FIELD, SHARED_FIELD, EXPIRY_FIELD = 1, 3, trials.FIELDS - 1


def rewrite_token_field(token: str, index: int, value) -> str:
    parts = token.split(".")
    parts[index] = str(value)
    return ".".join(parts)


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda t: None, id="absent"),
        pytest.param(lambda t: "", id="empty"),
        pytest.param(lambda t: "not-a-token", id="malformed"),
        pytest.param(lambda t: t[:-1] + ("a" if t[-1] != "a" else "b"), id="tampered-signature"),
        pytest.param(
            lambda t: rewrite_token_field(t, EXPIRY_FIELD, 99999999999), id="extended-expiry"
        ),
        # Every field is the server's claim about what it offered, including the
        # one that says an answer shouldn't be rated. Signing is what stops a
        # client deciding that for itself.
        pytest.param(lambda t: rewrite_token_field(t, SHARED_FIELD, 1), id="claimed-shared"),
    ],
)
def test_a_token_we_did_not_sign_is_refused(client, db, mangle):
    trial = next_trial(client)
    body = {**answer_body(trial), "trial_token": mangle(trial["trial_token"])}
    assert client.post("/api/answer", json=body).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


@pytest.fixture
def expired(monkeypatch):
    """Issue tokens that are already stale — a tab that sat through lunch."""
    monkeypatch.setattr(trials, "TOKEN_TTL_S", -1)
    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", -1)


def refresh(client, trial):
    r = client.post(
        "/api/trial/refresh",
        json={"item_id": trial["item_id"], "trial_token": trial["trial_token"]},
    )
    assert r.status_code == 200, r.text
    return {**trial, "trial_token": r.json()["trial_token"]}


def test_an_expired_token_is_refused_with_its_own_status(client, db, expired):
    """Distinct from the refusals a fresh token can't fix, because the client
    does something else with it: the trial is still this caller's, so the answer
    they already decided on is worth a round trip rather than a replacement
    position.

    And it costs the caller nothing on the way past — in particular no identity,
    which the refresh that follows would then find the token wasn't issued to.
    """
    trial = next_trial(client)
    r = client.post("/api/answer", json=answer_body(trial))
    assert r.status_code == 410
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert auth.COOKIE_NAME not in r.cookies


def test_a_token_that_aged_out_can_be_re_signed_and_answered(client, db, monkeypatch):
    """The whole point: the pick survives the tab having been left open."""
    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", -1)
    trial = next_trial(client)
    assert client.post("/api/answer", json=answer_body(trial)).status_code == 410

    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", 900)  # the clock we hand back
    answered = answer(client, refresh(client, trial))
    assert answered["correct"] in (True, False)
    assert db.execute("SELECT item_id FROM responses").fetchone()[0] == trial["item_id"]


def test_a_signed_in_tab_re_signs_its_trial_the_same_way(client, db, monkeypatch):
    """The other half of the same path: a bound token gets a longer life, not a
    different mechanism."""
    answer(client, next_trial(client))  # `client` now has an identity
    monkeypatch.setattr(trials, "TOKEN_TTL_S", -1)
    trial = next_trial(client)
    assert client.post("/api/answer", json=answer_body(trial)).status_code == 410

    monkeypatch.setattr(trials, "TOKEN_TTL_S", 12 * 3600)
    answer(client, refresh(client, trial))
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


def test_a_trial_past_its_own_deadline_is_finished_rather_than_stale(client, db, monkeypatch):
    """The deadline re-signing may not reach past. Refused as the kind of
    refusal there is no retry for, because a retry is exactly what would be
    wrong: this is where a trial stops existing, and the ledger that remembers
    whether it was spent stops having to."""
    trial = next_trial(client)
    monkeypatch.setattr(trials, "TRIAL_LIFE_S", -1)
    assert client.post("/api/answer", json=answer_body(trial)).status_code == 409
    assert (
        client.post(
            "/api/trial/refresh",
            json={"item_id": trial["item_id"], "trial_token": trial["trial_token"]},
        ).status_code
        == 409
    )
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_a_spent_anonymous_trial_is_remembered_until_it_can_no_longer_be_answered():
    """The coupling the whole re-signing story rests on, and the one a constant
    could silently break: a trial the ledger has forgotten must be one no token
    can still be redeemed against. Otherwise one held token is a fresh guest and
    a fresh first exposure against the same item, once per window, forever —
    which is the replay the ledger exists to stop.

    Reads the real limiter, so it takes none of the fixtures that swap it.
    """
    assert server.anonymous_trial_use.window_s >= trials.TRIAL_LIFE_S


def test_re_signing_says_nothing_about_the_item(client, expired):
    """It is on the pre-answer path, so it is held to the pre-answer rule: a
    caller who presents a token gets a token back and learns nothing they
    weren't already holding."""
    trial = next_trial(client)
    body = client.post(
        "/api/trial/refresh",
        json={"item_id": trial["item_id"], "trial_token": trial["trial_token"]},
    ).json()
    assert set(body) == {"item_id", "trial_token"}


def test_re_signing_carries_over_how_the_trial_was_served(client, db, monkeypatch):
    """Every field but the expiry, because a re-signed token is the same offer:
    a client that could shake off the shared mark by waiting out the clock would
    have the calibration exemption on demand."""
    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", -1)
    item = db.execute("SELECT id FROM items").fetchone()[0]
    trial = client.get(f"/api/next?item={item}").json()
    before = trial["trial_token"].split(".")
    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", 900)
    after = refresh(client, trial)["trial_token"].split(".")
    assert before[SHARED_FIELD] == after[SHARED_FIELD] == "1"
    assert before[:EXPIRY_FIELD] == after[:EXPIRY_FIELD]  # nonce included
    assert int(after[EXPIRY_FIELD]) > int(before[EXPIRY_FIELD])


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda t: None, id="absent"),
        pytest.param(lambda t: "not-a-token", id="malformed"),
        pytest.param(lambda t: t[:-1] + ("a" if t[-1] != "a" else "b"), id="tampered-signature"),
    ],
)
def test_only_a_token_we_can_read_is_re_signed(client, mangle):
    """A token we can't verify — forged, or signed with a key this process no
    longer holds — says nothing to carry over, and inventing the missing fields
    would let the caller choose them."""
    trial = next_trial(client)
    r = client.post(
        "/api/trial/refresh",
        json={"item_id": trial["item_id"], "trial_token": mangle(trial["trial_token"])},
    )
    assert r.status_code == 409


def test_re_signing_cannot_move_a_trial_to_another_session_or_item(client, db):
    """The two refusals that must survive it, since the answer that follows is
    filed under whoever presents the token: a pick made in one session may not be
    replayed into another, and a token is still for the one item it names."""
    answer(client, next_trial(client))  # `client` now has an identity
    mine = next_trial(client)
    with TestClient(server.app) as other:
        assert (
            other.post(
                "/api/trial/refresh",
                json={"item_id": mine["item_id"], "trial_token": mine["trial_token"]},
            ).status_code
            == 409
        )
    elsewhere = db.execute("SELECT id FROM items WHERE id != ?", (mine["item_id"],)).fetchone()[0]
    assert (
        client.post(
            "/api/trial/refresh",
            json={"item_id": elsewhere, "trial_token": mine["trial_token"]},
        ).status_code
        == 409
    )


def test_re_signing_cannot_re_arm_a_spent_anonymous_trial(client, db):
    """The ledger is keyed on the trial, not on the token naming it, so however
    many times one is re-signed they share the one slot. Otherwise refreshing
    would be a cheaper `/api/next` for a replayer: same item, no new exposure."""
    trial = next_trial(client)
    with TestClient(server.app) as spender:
        answer(spender, trial)
    with TestClient(server.app) as replayer:
        again = refresh(replayer, trial)
        assert replayer.post("/api/answer", json=answer_body(again)).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_an_anonymous_token_expires_sooner_than_a_bound_one(client):
    """It's the weaker kind — interchangeable between callers, and replayable
    unless the server still remembers it — so it gets a shorter life, which also
    keeps that memory small."""
    assert trials.ANON_TOKEN_TTL_S < trials.TOKEN_TTL_S
    anon = next_trial(client)["trial_token"]
    assert anon.split(".")[USER_FIELD] == "0"  # issued to nobody yet
    answer(client, next_trial(client))  # now `client` has an identity
    bound = next_trial(client)["trial_token"]
    assert bound.split(".")[USER_FIELD] != "0"
    assert int(bound.split(".")[EXPIRY_FIELD]) > int(anon.split(".")[EXPIRY_FIELD])


def test_an_answer_is_spent_once(client, db):
    """A re-answer is legitimate only once there is nothing fresh left, so
    anything else — a double tap, a retry after a slow response, a replayed
    token — is refused rather than recorded twice."""
    trial = next_trial(client)
    answer(client, trial)
    r = client.post("/api/answer", json=answer_body(trial))
    assert r.status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_answering_never_writes_to_the_item_bank(client, db):
    """Difficulty is a property of the position, so no answer — first exposure
    or repeat, right or wrong — may change what any other user is served: the
    `items` rows come out identical."""

    def items():
        return [tuple(row) for row in db.execute("SELECT * FROM items ORDER BY id")]

    before = items()
    for _ in range(2):  # exhaust the bank, so repeats become legitimate
        answer(client, next_trial(client))
    repeat = next_trial(client)
    assert repeat["repeat"] is True
    assert answer(client, repeat)["repeat"] is True

    assert items() == before
    # The responses are still recorded — they just aren't evidence about items.
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 3


def test_responses_carry_security_headers(client):
    """A CSP is what keeps a hostile string in mined game data from being
    script rather than a broken link; the rest is close-the-door hardening."""
    for path in ("/", "/api/account"):
        h = client.get(path).headers
        assert "default-src 'self'" in h["content-security-policy"]
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff"
        assert h["referrer-policy"] == "same-origin"


def csp_directives(response) -> dict[str, list[str]]:
    parts = response.headers["content-security-policy"].split(";")
    return {d.split()[0]: d.split()[1:] for d in (p.strip() for p in parts) if d}


# The source, not whichever tree is being served: minifying renames the
# constants these read, and what is being asserted is what the code we wrote
# does. The served copy is held to it separately, by the literals it still has
# to contain.
COUNT_JS = (server._ROOT / "web" / "count.js").read_text()
# count.js explains at length what GoatCounter's own script does instead, which
# names every field this one must not send. Whole-line comments only, which is
# all it has.
COUNT_JS_CODE = "\n".join(
    line for line in COUNT_JS.splitlines() if not line.strip().startswith("//")
)


def packed(text: str) -> str:
    """Whitespace out, so a literal reads the same minified as it does here."""
    return re.sub(r"\s+", "", text)


def count_js_endpoint() -> str:
    found = re.search(r'const ENDPOINT = "([^"]+)"', COUNT_JS)
    assert found, "count.js no longer declares an ENDPOINT this test can read"
    return found.group(1)


def counted_paths() -> dict[str, str]:
    """The beacon's whole vocabulary: URL path -> what it is reported as."""
    block = re.search(r"const COUNTED = \{(.*?)\n\};", COUNT_JS, re.S)
    assert block, "count.js no longer declares a COUNTED table this test can read"
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', block.group(1)))


@pytest.mark.parametrize("name", WEB_PAGES)
def test_every_page_counts_itself_and_loads_no_third_party_script(client, name):
    """The counter is a hosted service reached by our own code, so every page
    has to carry that code — and the page it reports has to be in the closed
    vocabulary, since anything outside it is silently uncounted."""
    path = "/" if name == "index.html" else f"/{name}"
    scripts = Head.of(client.get(path).text).scripts
    counters = [s for s in scripts if s.get("src", "").split("?")[0] == "count.js"]
    assert len(counters) == 1, f"{name} should load the counter exactly once"
    # Nothing here may be somebody else's: `script-src 'self'` is the point of
    # building the beacon ourselves, and an off-site src would need it widened.
    assert not [s for s in scripts if "//" in s.get("src", "")], f"{name} loads a foreign script"
    assert path in counted_paths(), f"{name} would be served but never counted"


def test_the_beacon_posts_where_the_csp_lets_it(client):
    """A CSP refusal is silent in the browser — the counter just stops counting
    — so the header and the endpoint are checked against each other."""
    endpoint = count_js_endpoint()
    assert endpoint == server.ANALYTICS_BEACON
    # And the copy that ships posts there too, whichever tree is being served.
    assert endpoint in client.get("/count.js").text
    csp = csp_directives(client.get("/"))
    assert csp["script-src"] == ["'self'"], "the counter needs no script origin"
    # `sendBeacon` is a connect; the fallback when that is refused is an image.
    assert endpoint in csp["connect-src"]
    assert endpoint in csp["img-src"]
    assert endpoint.startswith("https://")


def test_the_csp_allowlists_nothing_beyond_the_page_counter(client):
    """Enumerated, not grepped: a substring assertion still passes with
    `'unsafe-inline'` bolted on, which is how an allowlist rots."""
    allowed = {"'self'", "'none'", "data:", server.ANALYTICS_BEACON}
    for directive, sources in csp_directives(client.get("/")).items():
        assert set(sources) <= allowed, f"{directive} allows more than the counter"


def test_the_privacy_policy_names_the_counter_it_reports_to(client):
    """The counts land in somebody else's database, so the page that says what
    the site collects has to name them and link their terms."""
    policy = client.get("/privacy.html").text
    assert "GoatCounter" in policy
    assert "https://www.goatcounter.com/help/privacy" in policy


# What the beacon may send, and what each one is. GoatCounter's own script also
# sends `q` (the raw query string) and `t` (the title); both are absent here and
# that is the entire reason this file exists rather than a <script> tag.
BEACON_PARAMS = {"p", "s", "rnd", "r", "b"}


def test_the_counted_path_is_a_constant_and_never_the_url(client):
    """An item id in the URL is the research record, so a share link makes the
    query string exactly the thing that must not be reported. The defence is
    that the reported path is a value looked up in a table — a page missing
    from it counts as nothing, where a sanitizer would have counted as a leak.
    """
    vocabulary = counted_paths()
    assert vocabulary, "an empty table would pass every assertion below"
    shipped = packed(client.get("/count.js").text)
    for url_path, reported in vocabulary.items():
        assert reported in ("/", url_path), f"{url_path} reports as {reported}"
        # The build may only make files smaller, so the table it ships is the
        # same table — pairs and all.
        assert f'"{url_path}":"{reported}"' in shipped

    for leak in ("document.title", "location.search", "location.href", "location.hash"):
        assert leak not in COUNT_JS_CODE, f"{leak} would put the URL or the title in a hit"
    # The referrer is the second door to the same leak — from one of our pages
    # it is that page's full URL, item id and all — and the grep above doesn't
    # cover it, because `new URL(document.referrer).href` names neither
    # `location` nor `title`. So the function that reads it is held to
    # returning an origin.
    referrer_fn = COUNT_JS_CODE.split("function crossOriginReferrer()")[1].split("\n}")[0]
    assert "document.referrer" not in COUNT_JS_CODE.replace(referrer_fn, "")
    assert "url.origin" in referrer_fn
    # What it *returns*, not what it mentions: the substrings above still allow
    # `... ? null : document.referrer`, which is the whole leak in one line.
    returned = re.findall(r"return ([^;]+);", referrer_fn)
    assert returned, "the referrer function no longer returns in a form this can read"
    for expression in returned:
        assert "document.referrer" not in expression, f"reported verbatim: {expression}"
        assert "url" not in expression or "url.origin" in expression, expression
    for whole_url in (".href", ".toString()", ".pathname", ".search", "`${"):
        assert whole_url not in referrer_fn, f"{whole_url}: the referrer is more than an origin"
    sent = set(re.findall(r'params\.set\("(\w+)"', COUNT_JS_CODE)) | set(
        re.findall(r"^\s{4}(\w+):", COUNT_JS_CODE, re.M)
    )
    assert sent, "the parameters are no longer written in a form this test can read"
    assert sent <= BEACON_PARAMS, f"the beacon sends {sent - BEACON_PARAMS}"


def test_answering_is_rate_limited_but_arriving_is_free(client, db, monkeypatch):
    """Answering is the only unauthenticated write left, and the only thing that
    mints rows — so that's where the volume gate belongs. Arriving stays free,
    because a limit there is a gate in front of the first trial."""
    monkeypatch.setattr(server, "answer_limiter", auth.RateLimiter(1, 900))
    answer(client, next_trial(client))  # spends the only slot
    refused = client.post("/api/answer", json=answer_body(next_trial(client)))
    assert refused.status_code == 429
    # Reading is never refused, however many times.
    for _ in range(5):
        assert client.get("/api/next").status_code == 200
        assert client.get("/api/stats").status_code == 200
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_a_rate_limit_can_say_what_it_is_actually_rationing(client, monkeypatch):
    """The default wording speaks to someone who typed something wrong, which is
    not every limiter — so each carries its own message."""
    monkeypatch.setattr(server, "answer_limiter", auth.RateLimiter(1, 900, "a message of its own"))
    answer(client, next_trial(client))
    refused = client.post("/api/answer", json=answer_body(next_trial(client)))
    assert refused.json()["detail"] == "a message of its own"
    assert auth.RateLimiter(0, 900).message == auth.RateLimiter.DEFAULT_MESSAGE


def test_nothing_about_an_item_is_reflected_without_a_valid_token(client, db):
    """Not just the answer key: "that isn't one of the offered moves" tells an
    id-counting caller which two moves an item pairs, and a 404 tells them how
    big the bank is. So the token is checked before the item is even read."""
    trial = next_trial(client)
    other = db.execute("SELECT id FROM items WHERE id != ?", (trial["item_id"],)).fetchone()[0]
    for body in (
        {"item_id": other, "choice_uci": "e2e4"},  # no token at all
        {"item_id": other, "choice_uci": "h7h8", "trial_token": trial["trial_token"]},
        {"item_id": 99999, "choice_uci": "e2e4", "trial_token": trial["trial_token"]},
    ):
        r = client.post("/api/answer", json=body)
        assert r.status_code == 409, r.text
        assert "offered moves" not in r.text and "unknown item" not in r.text


def test_one_anonymous_token_cannot_be_replayed_into_many_first_exposures(client, db):
    """The case the ledger exists for — see the `trials` module docstring."""
    trial = next_trial(client)

    codes = []
    for _ in range(6):
        with TestClient(server.app) as fresh:  # cookieless every time
            codes.append(fresh.post("/api/answer", json=answer_body(trial)).status_code)

    assert codes == [200, 409, 409, 409, 409, 409]
    # One answer, not six — and so one identity minted, not six.
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_answering_your_last_unseen_item_does_not_make_its_token_replayable(client, db):
    """Answering the last unseen item takes the count to zero as a *result* of
    that answer, so a live count would call the token a legitimate repeat. The
    token says whether we served it as one instead.

    The first answer is spent getting an identity, so that the token under test is
    bound to a session — otherwise the binding refuses the replay first and this
    would pass without exercising the repeat rule at all.
    """
    answer(client, next_trial(client))
    last = next_trial(client)
    assert last["repeat"] is False
    answer(client, last)
    assert unanswered(db, client) == 0  # bank now empty

    for _ in range(3):
        assert client.post("/api/answer", json=answer_body(last)).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


@pytest.mark.parametrize("item_count", [1])
def test_a_repeat_we_served_stays_answerable_even_if_the_bank_refills(client, db, item_count):
    """The other side of the same boundary: a live count would turn a repeat the
    server had just offered into "you already answered this" the moment new
    items arrived mid-trial."""
    answer(client, next_trial(client))
    repeat = next_trial(client)
    assert repeat["repeat"] is True

    from .conftest import FEN_TMPL, add_item  # the bank grows under the open trial

    add_item(db, FEN_TMPL.format("7P"))
    db.commit()

    assert client.post("/api/answer", json=answer_body(repeat)).status_code == 200


def test_only_writing_ends_a_transaction_in_the_request_path():
    """One owner for every transaction, checked statically.

    A helper that commits ends its caller's transaction, which is invisible at
    the call site. The handle makes that unspellable and the guards catch it at
    runtime, but only on paths a test exercises; this covers the rest.
    """
    allowed = {"writing"}  # the one owner
    offenders = []
    # Inside a transaction, the ambient connection must not be named: `writing()`
    # hands out a handle with no commit on it, and reaching past that handle for
    # the module-level `conn` is how a block stops being one transaction.
    tree = ast.parse(Path(server.__file__).read_text())
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            is_writing = isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id", "") == "writing"
                for i in node.items
            )
            if not is_writing:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "conn":
                    offenders.append(f"server.py::{fn.name} names `conn` inside writing()")
    for path in (Path(server.__file__), Path(auth.__file__)):
        tree = ast.parse(path.read_text())
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if func.name in allowed:
                continue
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("commit", "rollback")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "conn"
                ):
                    offenders.append(f"{path.name}::{func.name} calls conn.{node.func.attr}()")
    assert not offenders, "only writing() may end a transaction; found " + "; ".join(offenders)


def test_a_statement_outside_a_transaction_stands_on_its_own(db):
    """The read paths group nothing, so a lone statement has to work."""
    server.conn.execute("SELECT 1")  # no transaction, no complaint


def test_the_ambient_connection_refuses_every_use_that_writing_should_own(db):
    """The rules that make a `writing()` block mean what it says.

    SQLite has no nested transaction, so a statement on the ambient connection
    while one is open — or a second `writing()` — can only end the first early.
    """
    # Ending a transaction isn't refused here, it's absent — and so is every
    # other route back to the raw connection that could have ended one.
    assert [name for name in dir(server.conn) if not name.startswith("_")] == ["execute"]

    # Reaching past the handle for the connection underneath.
    with pytest.raises(server.OutsideTransaction), server.writing():
        server.conn.execute("SELECT 1")

    # Opening a second transaction on top of one already open.
    with pytest.raises(server.OutsideTransaction), server.writing(), server.writing():
        pass


def test_a_failed_block_leaves_the_connection_usable(db):
    """A transaction left open is inherited by the next request on this thread,
    which could then never begin one."""
    with contextlib.suppress(ValueError), server.writing() as tx:
        tx.execute("SELECT 1")
        raise ValueError("boom")
    with server.writing() as tx:  # the next block still works
        tx.execute("SELECT 1")


def test_an_answer_that_waits_out_the_lock_is_told_to_retry(client, db, monkeypatch):
    """A write can lose the lock race, to another answer or to a bank refresh.
    The request was fine, so 500 — "don't bother trying again" — is the one
    thing that isn't true. Short timeout because the real one is ten seconds.
    """
    trial = next_trial(client)
    # The server opens its connections per thread, so the wait has to be short
    # before the one serving this request exists.
    monkeypatch.setattr(server.db, "BUSY_TIMEOUT_MS", 50)
    monkeypatch.setattr(server, "_threads", threading.local())

    holder = connect(server.DB_PATH, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE users SET rating = rating")  # holds the write lock
    try:
        r = client.post("/api/answer", json=answer_body(trial))
    finally:
        holder.rollback()
        holder.close()

    assert r.status_code == 503, r.text
    assert "busy" in r.json()["detail"].lower()
    # Nothing was recorded, so the retry is a clean first exposure.
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_a_real_database_error_is_still_a_500(client, monkeypatch):
    """Only SQLITE_BUSY is transient. A broken query is a bug, and telling the
    caller to retry it would just invite them to hammer."""

    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("no such column: nope")

    monkeypatch.setattr(server, "pick_item", broken)
    assert client.get("/api/next").status_code == 500


def test_a_signed_in_read_does_not_wait_on_a_writer(client, db, monkeypatch):
    """The read path refreshes a session's sliding expiry, and SQLite takes the
    write lock for an UPDATE whether or not a row matches. Left to the
    statement's own WHERE, a signed-in page load would fail behind a slow
    writer — which is a strange thing for it to depend on.
    """
    answer(client, next_trial(client))  # earns a session
    monkeypatch.setattr(server.db, "BUSY_TIMEOUT_MS", 50)
    monkeypatch.setattr(server, "_threads", threading.local())

    holder = connect(server.DB_PATH, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE users SET rating = rating")
    try:
        assert client.get("/api/next").status_code == 200
        assert client.get("/api/stats").status_code == 200
        assert client.get("/api/account").status_code == 200
    finally:
        holder.rollback()
        holder.close()


def test_assets_are_cached_forever_and_the_page_that_names_them_is_not(client):
    """The bargain: a URL that names its own contents can be kept forever, so
    the only thing a returning visitor has to ask about is the page itself."""
    page = client.get("/")
    assert page.headers["cache-control"] == assets.ENTRY_POINT

    referenced = re.findall(r'(?:src|href)="([^"]+\?v=[^"]+)"', page.text)
    assert referenced, "the page should reference versioned assets"
    for href in referenced:
        served = client.get("/" + href)
        assert served.status_code == 200, href
        assert served.headers["cache-control"] == assets.IMMUTABLE, href


def test_an_asset_reached_without_its_digest_is_not_pinned(client):
    """A bookmark or a crawler has no way to be told the file moved on, so it
    must not be handed a copy it will keep for a year."""
    assert client.get("/app.js").headers["cache-control"] == assets.UNVERSIONED
    assert client.get("/app.js?v=wrong").headers["cache-control"] == assets.UNVERSIONED


def test_a_digest_changes_when_the_file_does(tmp_path):
    """The whole scheme rests on this, and on it reaching *through* a file: the
    page's URL for app.js has to change when the module app.js imports does."""
    web = tmp_path / "web"
    (web / "vendor").mkdir(parents=True)
    (web / "vendor" / "lib.js").write_text("export const a = 1;\n")
    (web / "app.js").write_text('import { a } from "./vendor/lib.js";\n')
    (web / "index.html").write_text('<script src="app.js"></script>\n')

    before = assets.build(web)
    (web / "vendor" / "lib.js").write_text("export const a = 2;\n")
    after = assets.build(web)

    assert before["/vendor/lib.js"].digest != after["/vendor/lib.js"].digest
    assert before["/app.js"].digest != after["/app.js"].digest  # through the import
    assert before["/index.html"].body != after["/index.html"].body  # and into the page


def test_the_entry_point_still_revalidates_cheaply(client):
    """`no-cache` means asking every time, which is only cheap if the answer can
    be 304 rather than the page again."""
    first = client.get("/")
    again = client.get("/", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert not again.content


def fetch(client, path: str, accept_encoding: str):
    """A GET under one Accept-Encoding.

    The client decompresses what comes back, exactly as a browser does, so the
    body these assert on is the decoded one and `content-length` is what
    actually crossed the wire.
    """
    return client.get(path, headers={"Accept-Encoding": accept_encoding})


def test_a_client_that_takes_brotli_is_sent_brotli(client):
    """The whole reason compression is precomputed: the smallest copy is the
    default one, not something the server decides it can afford per request."""
    plain = fetch(client, "/app.js", "")
    sent = fetch(client, "/app.js", "br, gzip")
    assert sent.headers["content-encoding"] == "br"
    assert sent.content == plain.content  # decoded, so brotli round-tripped
    assert int(sent.headers["content-length"]) < len(plain.content) / 2


def test_a_client_that_refuses_brotli_is_sent_something_it_can_read(client):
    """`br;q=0` is a refusal, and is the reason the header is parsed rather than
    searched for a substring — answering in brotli anyway sends a body the
    client has just said it cannot decode."""
    sent = fetch(client, "/app.js", "br;q=0, gzip")
    assert sent.headers["content-encoding"] == "gzip"
    assert sent.content == fetch(client, "/app.js", "").content


def test_a_client_that_offers_nothing_is_sent_the_file_itself(client):
    plain = fetch(client, "/app.js", "")
    assert "content-encoding" not in plain.headers
    assert int(plain.headers["content-length"]) == len(plain.content)


def test_an_already_compressed_file_is_not_compressed_again(client):
    """A PNG gains nothing and would cost a decode on the way out."""
    assert "content-encoding" not in fetch(client, "/favicon-32x32.png", "br, gzip").headers


def test_a_compressed_variant_is_the_file_it_claims_to_be():
    """Checked against the bytes rather than through a client, because a client
    that decodes for you can't tell a wrong body from a right one."""
    built = assets.build(server.WEB_DIR)["/app.js"]
    assert brotli.decompress(built.encoded["br"]) == built.body
    assert gzip.decompress(built.encoded["gzip"]) == built.body


def test_every_asset_says_it_varies_by_encoding(client):
    """Including the ones with no variants to offer. A shared cache that stored
    one without `Vary` would hand it to a client that asked for something else,
    and which files have variants is not the client's business to track."""
    for path in ("/", "/app.js", "/style.css", "/favicon-32x32.png"):
        assert client.get(path).headers["vary"] == "Accept-Encoding", path


def test_two_encodings_of_one_file_do_not_share_a_tag(client):
    """An ETag names a body, and these are different bodies. Sharing one lets a
    cache answer 304 to a client holding the copy it can't read."""
    tags = {enc: fetch(client, "/app.js", enc).headers["etag"] for enc in ("br", "gzip", "")}
    assert len(set(tags.values())) == 3, tags
    # And the tag it does hand out is the one that comes back as a 304.
    for encoding, tag in tags.items():
        again = client.get("/app.js", headers={"Accept-Encoding": encoding, "If-None-Match": tag})
        assert again.status_code == 304, encoding


def test_the_build_changes_how_big_the_frontend_is_and_nothing_else():
    """web-dist/ is web/ made smaller and fewer — that is the whole contract
    with `scripts/build-web.mjs`, and it is what lets a dev checkout serve the
    sources and still be running the app the image serves.

    So every page has to come out the far side naming the same files. Bundling
    happens *below* that line: board.css swallows the three chessground
    stylesheets, and no page can tell.
    """
    built = server._ROOT / "web-dist"
    if not built.is_dir():
        pytest.skip("web-dist/ not built — `npm run build`")

    def references(tree: Path) -> dict[str, set[str]]:
        pattern = r'(?:src|href)="([^"]+)"'
        return {
            path: {ref.split("?")[0] for ref in re.findall(pattern, asset.body.decode())}
            for path, asset in assets.build(tree).items()
            if path.endswith(".html")
        }

    source, dist = references(server._ROOT / "web"), references(built)
    assert source == dist
