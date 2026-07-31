import itertools
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from trainer import auth, server
from trainer.db import connect

from .conftest import answer, answer_body, next_trial

CREDS = {"username": "tester", "password": "hunter2hunter2"}


def test_arriving_writes_nothing_and_answering_is_what_earns_an_identity(client, db):
    """Identity is issued by answering, not by arriving.

    An earlier version minted a guest row in the identity dependency, which made
    the cheapest write in the app the unauthenticated one — and metering that
    write is a limit in front of the first trial, which SPEC forbids. So a
    visitor who has answered nothing has no row, and the trial they are looking
    at rides in a signed token instead.
    """
    assert client.get("/api/account").json() == {"username": None, "guest": True}
    trial = next_trial(client)  # answerable immediately, nothing typed
    assert trial["item_id"] and trial["trial_token"]
    assert auth.COOKIE_NAME not in client.cookies
    assert row_counts(db) == (0, 0, 0)

    answer(client, trial)

    assert auth.COOKIE_NAME in client.cookies
    assert row_counts(db) == (1, 1, 1)
    assert client.get("/api/stats").json()["attempts"] == 1


def test_session_cookie_is_httponly_and_opaque(client):
    # The answer is what hands the cookie out, so that's the response to read it
    # from — nothing earlier in the flow sets one.
    r = client.post("/api/answer", json=answer_body(next_trial(client)))
    assert r.status_code == 200, r.text
    header = r.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "samesite=lax" in header.lower()
    # The identity is the token, and the token is never in a payload.
    assert client.cookies[auth.COOKIE_NAME] not in r.text


def test_signup_claims_the_guest_row_and_keeps_history(client, db):
    answer(client, next_trial(client))
    before = client.get("/api/stats").json()

    assert client.post("/api/account/signup", json=CREDS).json() == {
        "username": "tester",
        "guest": False,
    }

    after = client.get("/api/stats").json()
    assert after["attempts"] == before["attempts"] == 1
    assert after["user_rating"] == before["user_rating"]
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1  # claimed, not created


def test_signup_stores_optional_email_and_hashes_the_password(client, db):
    assert client.post("/api/account/signup", json={**CREDS, "email": "a@b.co"}).status_code == 200
    row = db.execute("SELECT * FROM users WHERE name = 'tester'").fetchone()
    assert row["email"] == "a@b.co"
    assert row["password_hash"].startswith("$argon2")
    assert CREDS["password"] not in row["password_hash"]


def test_signup_without_email_is_fine(client, db):
    assert client.post("/api/account/signup", json=CREDS).status_code == 200
    assert db.execute("SELECT email FROM users").fetchone()["email"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"username": "ab", "password": "hunter2hunter2"},  # too short
        {"username": "no spaces", "password": "hunter2hunter2"},
        {"username": "guest_abc", "password": "hunter2hunter2"},  # reserved prefix
        {"username": "tester", "password": "short"},
        {"username": "tester", "password": "hunter2hunter2", "email": "not-an-email"},
    ],
)
def test_signup_rejects_bad_input(client, body):
    r = client.post("/api/account/signup", json=body)
    assert r.status_code == 400
    assert r.json()["detail"]


def test_usernames_are_unique_case_insensitively(client):
    assert client.post("/api/account/signup", json=CREDS).status_code == 200
    with TestClient(server.app) as other:
        r = other.post("/api/account/signup", json={**CREDS, "username": "TESTER"})
        assert r.status_code == 400
        assert "taken" in r.json()["detail"]


def test_signup_is_refused_when_already_signed_in(client):
    client.post("/api/account/signup", json=CREDS)
    r = client.post("/api/account/signup", json={**CREDS, "username": "other"})
    assert r.status_code == 400


def test_login_restores_history_on_another_device(client, db):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)

    with TestClient(server.app) as other:
        assert other.get("/api/stats").json()["attempts"] == 0  # its own guest
        assert other.post("/api/account/login", json=CREDS).json()["username"] == "tester"
        assert other.get("/api/stats").json()["attempts"] == 1


def test_login_rejects_wrong_password_unknown_names_and_guest_rows(client, db):
    client.post("/api/account/signup", json=CREDS)
    with TestClient(server.app) as other:
        answer(other, next_trial(other))  # answering is what mints its row
        guest_name = db.execute("SELECT name FROM users WHERE password_hash IS NULL").fetchone()[0]
        for body in (
            {**CREDS, "password": "wrongwrongwrong"},
            {"username": "nobody", "password": "hunter2hunter2"},
            {"username": guest_name, "password": "hunter2hunter2"},
        ):
            r = other.post("/api/account/login", json=body)
            assert r.status_code == 400
            assert r.json()["detail"] == "Wrong username or password."


def test_logout_revokes_the_session_and_lands_on_a_fresh_guest(client, db):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    token = client.cookies[auth.COOKIE_NAME]

    assert client.post("/api/account/logout", json={}).status_code == 200
    assert auth.session_user(db, token) is None  # revoked server-side, not just cleared
    assert client.get("/api/account").json()["guest"] is True
    assert client.get("/api/stats").json()["attempts"] == 0


# --- deletion -------------------------------------------------------------
#
# The privacy policy promises that deleting an account takes its responses with
# it, so these tests are that promise: the rows are gone rather than merely
# detached, and only the password can trigger it.


def row_counts(conn) -> tuple[int, ...]:
    return tuple(
        conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("users", "sessions", "responses")
    )


def test_delete_erases_the_user_its_sessions_and_its_responses(client, db):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    token = client.cookies[auth.COOKIE_NAME]
    assert row_counts(db) == (1, 1, 1)

    r = client.post("/api/account/delete", json={"password": CREDS["password"]})
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "responses_deleted": 1}

    assert row_counts(db) == (0, 0, 0)  # erased, not detached
    assert auth.find_by_username(db, CREDS["username"]) is None
    assert auth.session_user(db, token) is None


def test_delete_clears_the_cookie_and_lands_on_a_fresh_guest(client):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)

    r = client.post("/api/account/delete", json={"password": CREDS["password"]})

    # Deleting the sessions row already revoked the token, so assert on the
    # header: without it this test would pass with the cookie clearing removed.
    assert "Max-Age=0" in r.headers["set-cookie"]
    assert auth.COOKIE_NAME not in client.cookies
    assert client.get("/api/account").json() == {"username": None, "guest": True}
    assert client.get("/api/stats").json()["attempts"] == 0
    # The name is free again, and claiming it inherits nothing from before.
    assert client.post("/api/account/signup", json=CREDS).status_code == 200
    assert client.get("/api/stats").json()["attempts"] == 0


def test_delete_needs_the_right_password(client, db):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)

    r = client.post("/api/account/delete", json={"password": "wrongwrongwrong"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Wrong password."
    assert row_counts(db) == (1, 1, 1)
    assert client.get("/api/account").json()["username"] == CREDS["username"]


def test_delete_guesses_are_throttled_like_logins(client, monkeypatch):
    """A shared browser holds the session, so the password is the only guard —
    and a password check with no limit on it is a guessing oracle."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "delete_limiter", auth.RateLimiter(2, 900))
    bad = {"password": "wrongwrongwrong"}
    assert client.post("/api/account/delete", json=bad).status_code == 400
    assert client.post("/api/account/delete", json=bad).status_code == 400
    # Out of tries, even with the right password.
    assert client.post("/api/account/delete", json=CREDS).status_code == 429


def test_a_login_lockout_cannot_block_deletion(client, monkeypatch):
    """Per-name login throttling means anyone can hold a known account locked
    by guessing at it. That must not reach the erase button: deletion is the
    privacy policy's promise, and it's the one thing a user might need to do
    urgently *because* someone is attacking the account."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    with TestClient(server.app) as attacker:
        for _ in range(3):
            attacker.post("/api/account/login", json=bad)
        # The owner really is locked out of signing in...
        assert attacker.post("/api/account/login", json=CREDS).status_code == 429
    # ...but the session they already have can still erase the account.
    r = client.post("/api/account/delete", json={"password": CREDS["password"]})
    assert r.status_code == 200, r.text


def test_a_guest_cannot_delete_and_keeps_its_history(client, db):
    """No password means nothing to authenticate against, and the cookie is the
    only handle on the row — so clearing it is what deletion means here."""
    answer(client, next_trial(client))

    r = client.post("/api/account/delete", json={"password": "anything at all"})
    assert r.status_code == 400
    assert "clearing the cookie" in r.json()["detail"]
    assert row_counts(db) == (1, 1, 1)


def test_a_cookieless_delete_mints_nothing(db):
    """A deletion request with no session has nothing to delete, so it must not
    write two rows on its way to being refused — the trap signup is shaped
    around, and the reason this endpoint resolves the session itself."""
    with TestClient(server.app) as cold:
        for _ in range(3):
            assert cold.post("/api/account/delete", json={"password": "x"}).status_code == 400
    assert row_counts(db) == (0, 0, 0)


def test_delete_leaves_other_users_alone(client, db):
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    with TestClient(server.app) as other:
        answer(other, next_trial(other))
        other.post("/api/account/signup", json={**CREDS, "username": "bystander"})

        client.post("/api/account/delete", json={"password": CREDS["password"]})

        assert row_counts(db) == (1, 1, 1)
        assert other.get("/api/stats").json()["attempts"] == 1


def test_delete_is_all_or_nothing(client, db):
    """Responses go before the user row they reference. A failure partway must
    not leave the account still standing with its answers already erased."""
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    user = auth.find_by_username(db, CREDS["username"])
    assert user is not None
    # Fail the second statement of the three, from inside SQLite.
    db.execute(
        "CREATE TRIGGER wedge BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'no'); END"
    )
    db.commit()

    # The caller owns the transaction — `delete_user` writes and nothing else,
    # which is what lets the endpoint commit it together with the cookie clear.
    with pytest.raises(sqlite3.IntegrityError), db:
        auth.delete_user(db, user["id"])

    db.execute("DROP TRIGGER wedge")
    db.commit()
    assert row_counts(db) == (1, 1, 1)  # rolled back whole, responses included


def test_deleting_an_account_frees_its_throttle_counter(client, monkeypatch):
    """SQLite reuses the highest id once its row goes, so a spent counter left
    behind would throttle whoever signs up next."""
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    client.post("/api/account/signup", json=CREDS)
    client.post("/api/account/delete", json={"password": "wrongwrongwrong"})
    client.post("/api/account/delete", json={"password": CREDS["password"]})

    client.post("/api/account/signup", json=CREDS)
    assert client.post("/api/account/delete", json=CREDS).status_code == 200


def test_repeated_wrong_passwords_are_throttled(client, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    with TestClient(server.app) as other:
        bad = {**CREDS, "password": "wrongwrongwrong"}
        assert other.post("/api/account/login", json=bad).status_code == 400
        assert other.post("/api/account/login", json=bad).status_code == 400
        # Right password, but out of tries.
        assert other.post("/api/account/login", json=CREDS).status_code == 429


def test_absurdly_long_password_is_refused_without_hashing_it(client):
    client.post("/api/account/signup", json=CREDS)
    with TestClient(server.app) as other:
        r = other.post(
            "/api/account/login", json={**CREDS, "password": "x" * (auth.MAX_PASSWORD_LEN + 1)}
        )
        assert r.status_code == 400


def test_signup_rate_limit_is_per_ip(db, monkeypatch):
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(1, 3600))
    with TestClient(server.app) as a, TestClient(server.app) as b:
        assert a.post("/api/account/signup", json=CREDS).status_code == 200
        assert b.post("/api/account/signup", json={**CREDS, "username": "sec"}).status_code == 429


def test_signup_limit_is_keyed_on_a_header_the_caller_cant_pick(db, monkeypatch):
    """Behind a proxy, the socket address is the caller's to choose.

    uvicorn's `--forwarded-allow-ips '*'` believes the leftmost
    `X-Forwarded-For` entry, and a proxy appends to what the client sent rather
    than replacing it — so a flooder who varies that header gets a fresh
    counter every request. Only a header the proxy overwrites can be charged.
    """
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(1, 3600))
    monkeypatch.setattr(server, "CLIENT_IP_HEADER", "fly-client-ip")

    def signup(client, username, real_ip, forwarded_lie):
        return client.post(
            "/api/account/signup",
            json={**CREDS, "username": username},
            headers={"Fly-Client-IP": real_ip, "X-Forwarded-For": forwarded_lie},
        ).status_code

    with TestClient(server.app) as a, TestClient(server.app) as b, TestClient(server.app) as c:
        assert signup(a, "one", "203.0.113.7", "10.0.0.1") == 200
        # Same real address, a fresh lie: one flooder, one counter.
        assert signup(b, "two", "203.0.113.7", "10.0.0.2") == 429
        # A genuinely different address is still unaffected — the limit has to
        # remain per-address, not become global.
        assert signup(c, "three", "198.51.100.4", "10.0.0.1") == 200


def test_rate_limiter_window_expires():
    limiter = auth.RateLimiter(2, window_s=60)
    limiter.consume("ip", now=0)
    limiter.consume("ip", now=1)
    with pytest.raises(auth.RateLimited):
        limiter.consume("ip", now=2)
    limiter.consume("ip", now=100)  # the first two have rolled out of the window


def test_rate_limiter_keys_are_independent():
    limiter = auth.RateLimiter(1, window_s=60)
    limiter.consume("a", now=0)
    limiter.consume("b", now=0)  # a different address is unaffected
    with pytest.raises(auth.RateLimited):
        limiter.consume("a", now=0)


def test_account_payloads_carry_no_credentials(client):
    client.post("/api/account/signup", json={**CREDS, "email": "a@b.co"})
    t = next_trial(client)
    body = (
        client.get("/api/account").text
        + client.get("/api/stats").text
        + client.get("/api/next").text
        + client.post(
            "/api/answer",
            json={"item_id": t["item_id"], "choice_uci": t["moves"][0]["uci"]},
        ).text
    )
    assert "a@b.co" not in body
    assert "argon2" not in body
    assert client.cookies[auth.COOKIE_NAME] not in body


def test_a_503_from_an_empty_bank_writes_nothing_at_all(tmp_path, monkeypatch):
    """This used to mint a guest per retry, which is why the cookie had to be
    re-applied from the error path — an orphaned row per retry otherwise. Now
    /api/next writes nothing, so there is nothing to orphan."""
    empty = connect(tmp_path / "empty.db", check_same_thread=False)
    monkeypatch.setattr(server, "conn", empty)
    with TestClient(server.app) as c:
        for _ in range(3):
            assert c.get("/api/next").status_code == 503
        assert auth.COOKIE_NAME not in c.cookies
    assert empty.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert empty.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize("item_count", [1])
def test_concurrent_answers_all_land(db, item_count):
    """Overlapping answers to one item from six sessions at once.

    Nothing about the *item* is contended any more — difficulty is fixed — but
    the six requests still share one `sqlite3.Connection` and so one implicit
    transaction: an unlocked `commit()` from one thread lands another's
    half-finished work, or its `rollback()` discards it. Hence the assertion
    that six answers produce six of everything rather than a correct tally.
    (One *session* can't have two answers in flight: /api/answer takes only the
    trial /api/next last served it, which is what keeps the answer key out of
    reach.)
    """
    clients = [TestClient(server.app) for _ in range(6)]
    try:
        trials = [next_trial(c) for c in clients]
        assert len({t["item_id"] for t in trials}) == 1  # one item, six answers
        threads = [
            threading.Thread(target=lambda c=c, t=t: answer(c, t))
            for c, t in zip(clients, trials, strict=True)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for c in clients:
            c.close()

    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 6
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 6
    # Each landed against the rating its own answer was scored under, so no
    # response carries a snapshot another thread's commit rolled forward.
    for row in db.execute("SELECT user_rating_before, user_rating_after FROM responses"):
        assert row["user_rating_before"] == server.rating.USER_START
        assert row["user_rating_after"] != row["user_rating_before"]


@pytest.mark.parametrize("item_count", [3])  # one to mint the identity, two to race
def test_two_tabs_answering_at_once_do_not_lose_a_rating_update(
    client, db, item_count, monkeypatch
):
    """One identity, two trials in flight, answered simultaneously.

    Trial tokens are signed rather than stored, so two tabs each hold a
    redeemable one. A rating is read-modify-write in Python, so without the
    write lock taken before the read both would start from the same value and
    the second would erase the first. `attempts` would still be right —
    SQLite resolves `attempts + 1` — so only the rating goes quietly wrong.
    """
    # Widen the read-modify-write window: at its real width the threads
    # interleave too rarely to catch a broken transaction at all.
    real_calibrate = server.rating.calibrate

    def slow_calibrate(*args, **kwargs):
        time.sleep(0.05)
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(server.rating, "calibrate", slow_calibrate)

    # Establish the identity first. Two trials fetched before one exists are
    # both issued to nobody, and answering either mints the row that makes the
    # other's token belong to someone else — a 409, and not the race under test.
    answer(client, next_trial(client))
    # Two *distinct* items: selection is a random pick among the nearest unseen
    # and nothing reserves one, so asking twice can name the same item.
    tabs = {}
    while len(tabs) < 2:
        t = next_trial(client)
        assert t["repeat"] is False
        tabs[t["item_id"]] = t
    threads = [threading.Thread(target=lambda t=t: answer(client, t)) for t in tabs.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    user = db.execute("SELECT rating, attempts FROM users").fetchone()
    assert user["attempts"] == 3  # the one that minted the row, plus the two raced
    # Both are in the record, and the second picked up where the first left off:
    # chain the snapshots and they have to meet in the middle.
    rows = list(
        db.execute("SELECT user_rating_before, user_rating_after FROM responses ORDER BY id")
    )
    assert len(rows) == 3
    for earlier, later in itertools.pairwise(rows):
        assert earlier["user_rating_after"] == later["user_rating_before"]
    assert rows[-1]["user_rating_after"] == user["rating"]


def test_garbage_cookie_falls_back_to_a_fresh_guest(client):
    client.cookies.set(auth.COOKIE_NAME, "not-a-real-token")
    r = client.get("/api/account")
    assert r.status_code == 200
    assert r.json()["guest"] is True


def test_expired_sessions_stop_working(client, db):
    client.post("/api/account/signup", json=CREDS)
    token = client.cookies[auth.COOKIE_NAME]
    assert auth.session_user(db, token) is not None
    db.execute(
        "UPDATE sessions SET last_seen = datetime('now', ?)", (f"-{auth.SESSION_DAYS + 1} days",)
    )
    db.commit()
    assert auth.session_user(db, token) is None
    assert client.get("/api/account").json()["guest"] is True  # not an error, just a new guest


def test_signup_rotates_the_session_token(client):
    answer(client, next_trial(client))  # the answer is what issues a session
    before = client.cookies[auth.COOKIE_NAME]
    client.post("/api/account/signup", json=CREDS)
    assert client.cookies[auth.COOKIE_NAME] != before


def test_sweep_drops_expired_sessions_but_never_users(client, db):
    answer(client, next_trial(client))
    db.execute("UPDATE sessions SET last_seen = datetime('now', '-400 days')")
    db.commit()

    auth.sweep(db)

    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_the_sweep_only_takes_sessions_no_request_could_use(db):
    """The sweep's conditions mirror `session_user`'s, so a session it deletes
    is one every request would already have refused — there is no window where
    a live identity gets pulled out from under its owner."""
    token = auth.start_session(db, auth.create_guest(db, 700, 250)["id"])
    db.execute("UPDATE sessions SET last_seen = datetime('now', '-364 days')")
    db.commit()

    auth.sweep(db)

    assert auth.session_user(db, token) is not None


def test_a_failed_answer_mints_no_identity(db, monkeypatch):
    """The guest row, its session, and the response commit together, so a
    failure while recording the answer rolls the identity back instead of
    leaving a row that answered nothing — which nothing sweeps."""

    def boom(*args):
        raise RuntimeError("simulated failure after the mint")

    monkeypatch.setattr(server.rating, "calibrate", boom)
    with TestClient(server.app, raise_server_exceptions=False) as c:
        trial = next_trial(c)
        assert c.post("/api/answer", json=answer_body(trial)).status_code == 500
    for table in ("users", "sessions", "responses"):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_a_session_can_never_point_at_a_missing_user(db):
    """The foreign key is enforced (SQLite leaves it off by default), which is
    what lets deletion rely on statement order and the sweep not chase
    orphans."""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO sessions (token_hash, user_id) VALUES ('x', 12345)")


def test_deleting_a_user_takes_its_sessions_but_never_its_responses(client, db):
    """Sessions cascade — worthless without their user — but responses
    deliberately don't: erasing the experimental record must be an explicit
    act (`delete_user`), not the side effect of a user row going away."""
    answer(client, next_trial(client))
    (user_id,) = db.execute("SELECT id FROM users").fetchone()

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))

    db.execute("DELETE FROM responses WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_sweep_keeps_claimed_accounts_even_when_idle(client, db):
    client.post("/api/account/signup", json=CREDS)
    db.execute("UPDATE users SET created_at = datetime('now', '-400 days')")
    db.execute("UPDATE sessions SET last_seen = datetime('now', '-400 days')")
    db.commit()

    auth.sweep(db)

    assert auth.find_by_username(db, "tester") is not None


def test_api_responses_are_never_cacheable(client):
    for path in ("/api/account", "/api/stats", "/api/next"):
        r = client.get(path)
        assert r.headers["cache-control"] == "no-store", path
        assert r.headers["vary"] == "Cookie", path


def test_failed_signups_are_throttled(client, monkeypatch):
    """Rejections must cost something: each one buys an argon2 hash and an
    answer to 'does this username exist?'."""
    taken = {**CREDS, "username": "someoneelse"}
    with TestClient(server.app) as owner:
        owner.post("/api/account/signup", json=taken)
    # Patched after the setup signup: TestClient reports one host for every
    # client, so they all share a limiter key.
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(3, 3600))
    for _ in range(3):
        assert client.post("/api/account/signup", json=taken).status_code == 400
    assert client.post("/api/account/signup", json=taken).status_code == 429
    # ...and a valid signup is refused too, rather than slipping past the gate.
    assert client.post("/api/account/signup", json=CREDS).status_code == 429


def test_taken_username_is_rejected_without_hashing(client, monkeypatch):
    with TestClient(server.app) as owner:
        owner.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(
        auth, "hash_password", lambda _: pytest.fail("hashed before checking the name")
    )
    assert client.post("/api/account/signup", json=CREDS).status_code == 400


def test_concurrent_login_guesses_cannot_outrun_the_limit(client, monkeypatch):
    """check-then-record with slow work in between lets every in-flight
    request pass a counter none of them has incremented yet."""
    client.post("/api/account/signup", json=CREDS)
    limit = 5
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(limit, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    codes = []
    with TestClient(server.app) as other:
        threads = [
            threading.Thread(
                target=lambda: codes.append(other.post("/api/account/login", json=bad).status_code)
            )
            for _ in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    # The limiter admits exactly `limit`, however many arrive at once. Those
    # it admits either guess wrong (400) or find the hasher saturated and get
    # shed (503) — a burst this size is exactly what the shedding is for.
    admitted = [c for c in codes if c != 429]
    assert len(admitted) == limit, codes
    assert set(admitted) <= {400, 503}, codes


def test_rate_limiter_consume_is_atomic_under_threads():
    limiter = auth.RateLimiter(10, window_s=3600)
    allowed = []

    def attempt():
        try:
            limiter.consume("ip")
            allowed.append(1)
        except auth.RateLimited:
            pass

    threads = [threading.Thread(target=attempt) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 10


def test_a_crash_while_answering_still_hands_out_the_identity_it_created(db, monkeypatch):
    """The identity is now created while serving the *answer*, so that's where
    this invariant lives. A 500 is handled outside our middleware, and the
    answer's row is already committed by the time the reveal is built — if the
    error response drops the Set-Cookie, the row is orphaned and the next answer
    mints another one."""

    def boom(*_args):
        raise RuntimeError("boom")

    with TestClient(server.app, raise_server_exceptions=False) as c:
        trial = next_trial(c)
        monkeypatch.setattr(server, "line_steps", boom)  # after the writes commit
        assert c.post("/api/answer", json=answer_body(trial)).status_code == 500
        assert auth.COOKIE_NAME in c.cookies
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_throttled_signups_do_not_mint_guest_rows(db, monkeypatch):
    """Identity is resolved after the gate, not by a dependency in front of
    it — otherwise a rejected flood leaves one row per attempt behind."""
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(0, 3600))
    for _ in range(5):
        with TestClient(server.app) as c:
            assert c.post("/api/account/signup", json=CREDS).status_code == 429
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_saturated_hasher_sheds_load_instead_of_queueing(client, monkeypatch):
    """Blocking forever on the argon2 semaphore just trades an out-of-memory
    for a stalled thread pool, with the trial flow stuck behind the queue."""
    monkeypatch.setattr(auth, "HASH_WAIT_S", 0.05)
    monkeypatch.setattr(auth, "_hash_slots", threading.Semaphore(0))  # fully saturated
    r = client.post("/api/account/signup", json=CREDS)
    assert r.status_code == 503
    assert "try again" in r.json()["detail"].lower()


def test_rate_limiter_key_space_is_bounded():
    limiter = auth.RateLimiter(5, window_s=3600)
    monkey = limiter.MAX_KEYS + 5000
    for i in range(monkey):
        limiter.consume(f"ip-{i}", now=float(i))
    assert len(limiter._hits) <= limiter.MAX_KEYS


def test_a_shed_signup_still_costs_a_slot(db, monkeypatch):
    """Refunding a shed attempt would leave a saturated box with an unmetered
    signup endpoint — worse than the throttling it avoids. (It no longer leaves
    a row behind either: the hash is refused before anything is written.)"""
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(3, 3600))
    monkeypatch.setattr(auth, "HASH_WAIT_S", 0.05)
    monkeypatch.setattr(auth, "_hash_slots", threading.Semaphore(0))
    codes = []
    for _ in range(6):
        with TestClient(server.app) as c:  # a fresh browser each time
            codes.append(c.post("/api/account/signup", json=CREDS).status_code)
    assert codes == [503, 503, 503, 429, 429, 429], codes
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0  # nothing written


def test_every_signup_attempt_counts_including_rejected_ones(client, monkeypatch):
    """No refunds anywhere: a request that fails validation, hits a taken
    name, or gets shed still spends its slot. That's what stops a flood from
    buying unlimited argon2 and unlimited "is this name taken?" answers. The
    limit is loose instead of clever, so a person fumbling the form never
    reaches it."""
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(3, 3600))
    assert client.post("/api/account/signup", json={**CREDS, "password": "sh"}).status_code == 400
    assert client.post("/api/account/signup", json={**CREDS, "username": "!"}).status_code == 400
    assert client.post("/api/account/signup", json=CREDS).status_code == 200
    with TestClient(server.app) as other:
        assert (
            other.post("/api/account/signup", json={**CREDS, "username": "x2"}).status_code == 429
        )


def test_guessing_is_throttled_per_account_not_per_address(client, monkeypatch):
    """The thing being attacked is an account, so that's what we count. An
    attacker rotating addresses — one line of script — must not get a fresh
    budget each time, which is exactly what a per-IP limit would give them."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(3, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    codes = []
    for i in range(8):
        monkeypatch.setattr(server, "client_key", lambda _r, i=i: f"10.0.0.{i}")
        with TestClient(server.app) as attacker:  # a new address every time
            codes.append(attacker.post("/api/account/login", json=bad).status_code)
    assert codes.count(400) == 3, codes
    assert codes.count(429) == 5, codes


def test_throttling_one_account_does_not_touch_another(client, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    with TestClient(server.app) as second:
        second.post("/api/account/signup", json={"username": "other", "password": "hunter2hunter2"})
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    with TestClient(server.app) as c:
        for _ in range(2):
            assert c.post("/api/account/login", json=bad).status_code == 400
        assert c.post("/api/account/login", json=bad).status_code == 429  # tester is out
        # ...but the other account, from the same address, is unaffected.
        r = c.post("/api/account/login", json={"username": "other", "password": "hunter2hunter2"})
        assert r.status_code == 200


def test_unknown_usernames_do_not_consume_a_real_accounts_budget(client, monkeypatch):
    """Every name gets its own counter, so junk names can't spend the budget
    that protects a real account."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    with TestClient(server.app) as c:
        for i in range(20):
            r = c.post("/api/account/login", json={"username": f"nobody{i}", "password": "x" * 12})
            assert r.status_code == 400
        assert c.post("/api/account/login", json=CREDS).status_code == 200  # budget intact


def test_an_unknown_username_is_throttled_exactly_like_a_real_one(client, monkeypatch):
    """Otherwise the throttle itself answers "does this account exist?": a 429
    for a name that has a counter, 400 forever for one that doesn't. Eleven
    requests and no credentials, and it undoes the dummy-hash verify that keeps
    the *timing* from saying the same thing."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    bad = {"password": "wrongwrongwrong"}
    with TestClient(server.app) as c:
        real = [
            c.post("/api/account/login", json={**bad, "username": CREDS["username"]}).status_code
        ]
        real.append(
            c.post("/api/account/login", json={**bad, "username": CREDS["username"]}).status_code
        )
        real.append(
            c.post("/api/account/login", json={**bad, "username": CREDS["username"]}).status_code
        )
        ghost = [
            c.post("/api/account/login", json={**bad, "username": "nosuchperson"}).status_code
            for _ in range(3)
        ]
    assert real == [400, 400, 429]
    assert ghost == real  # the boundary says nothing about which names exist


def test_login_is_capped_per_address_even_for_names_that_do_not_exist(client, monkeypatch):
    """argon2 is 64 MiB and ~50ms by design and only HASH_CONCURRENCY of them
    run at once, so an unmetered password check is a way to answer every real
    user 503. A name nobody registered must still cost the caller something."""
    monkeypatch.setattr(server, "login_ip_limiter", auth.RateLimiter(3, 900))
    with TestClient(server.app) as c:
        codes = [
            c.post(
                "/api/account/login", json={"username": f"ghost{i}", "password": "x" * 12}
            ).status_code
            for i in range(5)
        ]
    assert codes == [400, 400, 400, 429, 429]


def test_a_correct_password_clears_the_accounts_strikes(client, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(3, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    with TestClient(server.app) as c:
        for _ in range(2):
            assert c.post("/api/account/login", json=bad).status_code == 400
        assert c.post("/api/account/login", json=CREDS).status_code == 200
        # Strikes reset, so a user who fumbled and then got it right isn't
        # left one typo from a lockout.
        for _ in range(3):
            assert c.post("/api/account/login", json=bad).status_code == 400


def test_a_saturated_hasher_waits_briefly_then_sheds(client, monkeypatch):
    """The wait bounds memory without trading it for a stalled thread pool:
    sync endpoints run on a fixed pool, so a caller queueing indefinitely
    holds a thread the trial flow needs."""
    assert auth.HASH_WAIT_S <= 0.5  # the bound the trial flow relies on
    monkeypatch.setattr(auth, "_hash_slots", threading.Semaphore(0))
    started = time.monotonic()
    r = client.post("/api/account/signup", json=CREDS)
    waited = time.monotonic() - started
    assert r.status_code == 503
    assert "try again" in r.json()["detail"].lower()
    assert waited < auth.HASH_WAIT_S + 1


def test_setting_a_password_signs_existing_sessions_out(client, db):
    """`trainer.account set-password` is the only recovery path this app has —
    no in-app password change, no reset email — so it has to assume the reason
    it's being run is that someone else knows the old password. Rotating the
    hash while leaving their session live recovers nothing."""
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    token = client.cookies[auth.COOKIE_NAME]
    assert auth.session_user(db, token) is not None

    user = auth.find_by_username(db, CREDS["username"])
    assert user is not None
    db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (auth.hash_password("a-brand-new-password"), user["id"]),
    )
    db.commit()
    with db:  # helpers write; committing is the caller's job
        assert auth.revoke_sessions(db, user["id"]) == 1

    assert auth.session_user(db, token) is None
    assert client.get("/api/account").json()["guest"] is True
    # The history is still there for whoever knows the new password.
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_a_session_expires_absolutely_however_often_it_is_used(client, db):
    """The idle window slides forward on every request, so on its own it never
    expires a token that gets used — a stolen cookie would be permanent."""
    client.post("/api/account/signup", json=CREDS)
    token = client.cookies[auth.COOKIE_NAME]
    assert auth.session_user(db, token) is not None

    db.execute(
        "UPDATE sessions SET created_at = datetime('now', ?), last_seen = datetime('now')",
        (f"-{auth.SESSION_MAX_DAYS + 1} days",),
    )
    db.commit()
    assert auth.session_user(db, token) is None  # warm, but too old to matter


def test_an_ambiguous_username_is_refused_rather_than_guessed(db):
    """Only reachable on a database whose case-insensitive unique index couldn't
    be created. Picking one silently is how `set-password kim` ends up setting
    it on `Kim`'s row, handing one user another's account."""
    db.execute("DROP INDEX IF EXISTS idx_users_name_nocase")
    for name in ("Kim", "kim"):
        db.execute("INSERT INTO users (name, rating, calib_step) VALUES (?, 700, 250)", (name,))
    db.commit()
    with pytest.raises(auth.AuthError, match="matches 2 rows"):
        auth.find_by_username(db, "KIM")


def test_the_limiter_forgives_the_least_throttled_key_first(monkeypatch):
    """Key eviction is unavoidable at the cap, but which key goes matters: a
    login key is whatever username the caller typed, so evicting by age lets a
    flood of one-hit junk push out the near-exhausted counter of the account
    being guessed at and hand the attacker a fresh budget."""
    limiter = auth.RateLimiter(limit=3, window_s=900)
    monkeypatch.setattr(type(limiter), "MAX_KEYS", 8)
    for _ in range(3):
        limiter.consume("name:victim", now=1.0)
    with pytest.raises(auth.RateLimited):
        limiter.consume("name:victim", now=1.0)

    for i in range(40):  # a flood of freshly-minted, barely-used keys
        limiter.consume(f"name:junk{i}", now=2.0)

    assert "name:victim" in limiter._hits  # survived; still out of guesses
    with pytest.raises(auth.RateLimited):
        limiter.consume("name:victim", now=2.0)


def test_signing_up_before_answering_anything_creates_the_row(client, db):
    """There's no guest row to claim until something has been answered, so the
    drawer has to work for someone who opened it first thing."""
    assert row_counts(db) == (0, 0, 0)
    assert client.post("/api/account/signup", json=CREDS).json() == {
        "username": "tester",
        "guest": False,
    }
    assert row_counts(db) == (1, 1, 0)  # a real account, no history yet
    assert client.get("/api/account").json()["username"] == "tester"
    # And it's a working account: answers land on it, and it survives a re-login.
    answer(client, next_trial(client))
    with TestClient(server.app) as other:
        assert other.post("/api/account/login", json=CREDS).status_code == 200
        assert other.get("/api/stats").json()["attempts"] == 1


def test_a_pre_identity_trial_is_spent_once_however_many_contexts_use_it(db):
    """A token issued before its holder has an identity can't be bound to a
    session that doesn't exist yet, so it is spent once and remembered instead.

    The two clients below are one person's two contexts — a browser and a script —
    not two people: a token goes to the one client that asked for it and nowhere
    else, so nobody else has it to try. What that buys isn't privacy of the answer
    key (the reveal gives that up the moment you commit) but a spent trial staying
    spent: an authenticated replay is caught by the `responses` row it already
    wrote, and each anonymous replay mints a fresh identity, so that row is never
    there to find and the ledger has to stand in for it.
    """
    with TestClient(server.app) as browser, TestClient(server.app) as script:
        trial = next_trial(browser)
        peeked = script.post("/api/answer", json=answer_body(trial))
        assert peeked.status_code == 200  # unbound, so either context may spend it
        assert peeked.json()["best"]["uci"]

        # Spent — by two different mechanisms, both reported as the status the
        # client recovers from. The script now holds a session the anonymous
        # token isn't bound to; the still-anonymous browser meets the ledger.
        assert script.post("/api/answer", json=answer_body(trial)).status_code == 409
        assert browser.post("/api/answer", json=answer_body(trial)).status_code == 409
        # So the peek can't be banked as a correct answer either.
        assert browser.get("/api/stats").json()["attempts"] == 0
        answer(browser, next_trial(browser))  # a fresh trial works fine

    # One row and one response per context — no per-replay inflation.
    assert row_counts(db) == (2, 2, 2)
