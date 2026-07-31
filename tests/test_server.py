import ast
import contextlib
import sqlite3
import struct
import threading
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainer import auth, server, trials
from trainer.db import connect

from .conftest import ITEM, answer, answer_body, next_trial


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
    assert client.get("/api/stats").json()["items_remaining"] == 0
    rating_before = user_row(db, client)["rating"]
    result = answer(client, t)
    assert result["repeat"] is True
    assert "correct" in result  # feedback still shown
    assert user_row(db, client)["rating"] == rating_before  # but no rating movement


def test_first_exposure_accuracy_excludes_repeats(client):
    """Repeats are answerable from memory of the reveal, so they say nothing
    about skill and must not move the reported accuracy either way."""

    def answer_with(trial, uci):
        answer(client, trial, [m["uci"] for m in trial["moves"]].index(uci))

    for _ in range(2):  # the whole bank, answered correctly
        answer_with(next_trial(client), ITEM["best_uci"])
    assert client.get("/api/stats").json()["accuracy_last_50"] == 1.0

    for _ in range(2):  # now repeats, answered wrongly
        t = next_trial(client)
        assert t["repeat"] is True
        answer_with(t, ITEM["distractor_uci"])
    stats = client.get("/api/stats").json()
    assert stats["attempts"] == 4  # all four were recorded
    assert stats["accuracy_last_50"] == 1.0  # but only the two fresh ones counted


def test_first_exposure_filter_is_answered_from_an_index_covering_item_id(db):
    """The filter asks, per response, whether an earlier one hit the same item.
    Without `item_id` indexed that walks every earlier row the user has, once
    per row: 700ms against 3ms at 5k responses.

    Asserts the plan, not a duration, because a timing threshold flakes on CI.
    The losing plan is a SEARCH too — over a range rather than a row — so which
    index gets chosen is the whole assertion.
    """
    plan = db.execute("EXPLAIN QUERY PLAN " + server.RECENT_FIRST_EXPOSURES_SQL, (1,)).fetchall()
    inner = [row[-1] for row in plan if "p" in row[-1].split()]
    assert inner, f"no plan step for the inner query: {[r[-1] for r in plan]}"
    assert "idx_responses_item" in inner[0], inner[0]


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


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda t: None, id="absent"),
        pytest.param(lambda t: "", id="empty"),
        pytest.param(lambda t: "not-a-token", id="malformed"),
        pytest.param(lambda t: t[:-1] + ("a" if t[-1] != "a" else "b"), id="tampered-signature"),
        pytest.param(lambda t: t.replace(t.split(".")[2], "99999999999", 1), id="extended-expiry"),
    ],
)
def test_a_token_we_did_not_sign_is_refused(client, db, mangle):
    trial = next_trial(client)
    body = {**answer_body(trial), "trial_token": mangle(trial["trial_token"])}
    assert client.post("/api/answer", json=body).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_an_expired_token_is_refused(client, db, monkeypatch):
    """A tab left open for a day should ask for a fresh trial, not answer a
    stale one — the client turns the 409 into exactly that."""
    monkeypatch.setattr(trials, "TOKEN_TTL_S", -1)
    monkeypatch.setattr(trials, "ANON_TOKEN_TTL_S", -1)
    trial = next_trial(client)
    assert client.post("/api/answer", json=answer_body(trial)).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_an_anonymous_token_expires_sooner_than_a_bound_one(client):
    """It's the weaker kind — interchangeable between callers, and replayable
    unless the server still remembers it — so it gets a shorter life, which also
    keeps that memory small."""
    assert trials.ANON_TOKEN_TTL_S < trials.TOKEN_TTL_S
    anon = next_trial(client)["trial_token"]
    assert anon.split(".")[1] == "0"  # the user field: issued to nobody yet
    answer(client, next_trial(client))  # now `client` has an identity
    bound = next_trial(client)["trial_token"]
    assert bound.split(".")[1] != "0"
    assert int(bound.split(".")[4]) > int(anon.split(".")[4])


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
    or repeat, right or wrong — may change what any other user is served. That
    guarantee is now the whole of it: the `items` rows come out identical."""

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
    """The case the ledger exists for. Redeeming an anonymous token *creates*
    the identity that records it, so every replay would be a brand new row
    seeing the item for the first time — and the `responses` row that makes a
    spent trial unanswerable for an authenticated caller is never there to find.
    The ledger is what stands in for it."""
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
    """The boundary a live `unseen_count` check got wrong: answering the last
    unseen item takes the count to zero as a *result* of that answer, which made
    "is a repeat allowed right now?" true and the token replayable. The token
    itself now says whether we served it as a repeat.

    The first answer is spent getting an identity, so that the token under test is
    bound to a session — otherwise the binding refuses the replay first and this
    would pass without exercising the repeat rule at all.
    """
    answer(client, next_trial(client))
    last = next_trial(client)
    assert last["repeat"] is False
    answer(client, last)
    assert client.get("/api/stats").json()["items_remaining"] == 0  # bank now empty

    for _ in range(3):
        assert client.post("/api/answer", json=answer_body(last)).status_code == 409
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


@pytest.mark.parametrize("item_count", [1])
def test_a_repeat_we_served_stays_answerable_even_if_the_bank_refills(client, db, item_count):
    """The mirror of the same bug: a live count would turn a repeat the server
    had just offered into "you already answered this" the moment new items
    arrived mid-trial."""
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
