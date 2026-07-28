import threading
import time

import pytest
from fastapi.testclient import TestClient

from trainer import auth, server
from trainer.db import connect

from .conftest import answer, next_trial

CREDS = {"username": "tester", "password": "hunter2hunter2"}


def test_guest_identity_needs_no_signup(client):
    assert client.get("/api/account").json() == {"username": None, "guest": True}
    assert auth.COOKIE_NAME in client.cookies
    assert next_trial(client)["item_id"]  # answerable immediately, nothing typed


def test_session_cookie_is_httponly_and_opaque(client):
    r = client.get("/api/account")
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
        other.get("/api/account")  # mints a guest row for `other`
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


def test_signup_typos_do_not_burn_attempts(client, monkeypatch):
    """A user fumbling the form must not lock themselves out of signing up."""
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(1, 3600))
    for bad in ({**CREDS, "password": "short"}, {**CREDS, "username": "!!"}):
        assert client.post("/api/account/signup", json=bad).status_code == 400
    assert client.post("/api/account/signup", json=CREDS).status_code == 200


def test_successful_logins_hand_their_slot_back(client, monkeypatch):
    """The slot is held across the verify — so a burst can't walk past the
    limit — but a correct password refunds it, or everyone behind one NAT
    would throttle each other just by signing in."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(1, 900))
    for _ in range(3):
        with TestClient(server.app) as other:
            assert other.post("/api/account/login", json=CREDS).status_code == 200


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


def test_failed_request_still_hands_out_the_identity_it_created(tmp_path, monkeypatch):
    """A 503 from an empty bank must not mint an orphan row per retry.

    The guest is committed while serving the request; if the error response
    drops its Set-Cookie, the client never gets an identity and every retry
    creates another unreachable row.
    """
    empty = connect(tmp_path / "empty.db", check_same_thread=False)
    monkeypatch.setattr(server, "conn", empty)
    with TestClient(server.app) as c:
        for _ in range(3):
            assert c.get("/api/next").status_code == 503
        assert auth.COOKIE_NAME in c.cookies
    assert empty.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert empty.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_concurrent_answers_do_not_clobber_rating(client, db):
    """Two overlapping answers on one session (two tabs, or a retried POST).

    The user row is read-modify-written, so both requests must serialize on
    the same lock the write happens under — a row read before that lock is a
    snapshot, and the second writer would roll the first one back.
    """
    trials = [next_trial(client) for _ in range(1)]
    trials.append(next_trial(client))
    results = []
    threads = [
        threading.Thread(target=lambda t=t: results.append(answer(client, t))) for t in trials
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    row = db.execute("SELECT rating, calib_step, attempts FROM users").fetchone()
    assert row["attempts"] == 2
    # Both were correct-or-not, but each moved the rating from the previous
    # one's result: the final row must equal the last recorded snapshot.
    last = db.execute(
        "SELECT user_rating_after FROM responses ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    assert row["rating"] == last
    assert db.execute("SELECT COUNT(DISTINCT user_rating_before) FROM responses").fetchone()[0] == 2


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
    client.get("/api/account")
    before = client.cookies[auth.COOKIE_NAME]
    client.post("/api/account/signup", json=CREDS)
    assert client.cookies[auth.COOKIE_NAME] != before


def test_sweep_drops_empty_guests_but_never_history(client, db):
    answer(client, next_trial(client))  # this guest has a response
    with TestClient(server.app) as idle:
        idle.get("/api/account")  # this one has nothing
    db.execute("UPDATE users SET created_at = datetime('now', '-30 days')")
    db.execute("UPDATE sessions SET last_seen = datetime('now', '-30 days')")
    db.commit()

    auth.sweep(db)

    names = [r[0] for r in db.execute("SELECT name FROM users")]
    assert len(names) == 1  # the idle guest is gone
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


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
    monkeypatch.setattr(server, "signup_attempt_limiter", auth.RateLimiter(3, 3600))
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
    assert codes.count(400) <= limit, codes
    assert codes.count(429) == 20 - codes.count(400)


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


def test_crash_still_hands_out_the_identity_it_created(client, db, monkeypatch):
    """A 500 is handled outside our middleware; the cookie has to survive it."""
    monkeypatch.setattr(server, "pick_item", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    with TestClient(server.app, raise_server_exceptions=False) as c:
        for _ in range(3):
            assert c.get("/api/next").status_code == 500
        assert auth.COOKIE_NAME in c.cookies
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_throttled_signups_do_not_mint_guest_rows(db, monkeypatch):
    """Identity is resolved after the gate, not by a dependency in front of
    it — otherwise a rejected flood leaves one row per attempt behind."""
    monkeypatch.setattr(server, "signup_attempt_limiter", auth.RateLimiter(0, 3600))
    for _ in range(5):
        with TestClient(server.app) as c:
            assert c.post("/api/account/signup", json=CREDS).status_code == 429
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_concurrent_signups_cannot_outrun_the_creation_limit(db, monkeypatch):
    """The creation limit is the one that says how many accounts an address
    gets; checking it before the ~50ms hash and recording after lets every
    request in flight pass a counter none of them has incremented."""
    limit = 3
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(limit, 3600))
    monkeypatch.setattr(server, "signup_attempt_limiter", auth.RateLimiter(50, 3600))
    codes = []
    barrier = threading.Barrier(15)

    def attempt(i):
        with TestClient(server.app) as c:
            barrier.wait()
            codes.append(
                c.post("/api/account/signup", json={**CREDS, "username": f"u{i}"}).status_code
            )

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes.count(200) == limit, codes
    created = db.execute("SELECT COUNT(*) FROM users WHERE password_hash IS NOT NULL").fetchone()[0]
    assert created == limit


def test_saturated_hasher_sheds_load_instead_of_queueing(client, monkeypatch):
    """Blocking forever on the argon2 semaphore just trades an out-of-memory
    for a stalled thread pool, with the trial flow stuck behind the queue."""
    monkeypatch.setattr(auth, "HASH_WAIT_S", 0.05)
    monkeypatch.setattr(auth, "_hash_slots", threading.Semaphore(0))  # fully saturated
    r = client.post("/api/account/signup", json=CREDS)
    assert r.status_code == 503
    assert "try again" in r.json()["detail"].lower()


def test_a_shed_login_does_not_spend_a_slot(client, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(1, 900))
    monkeypatch.setattr(auth, "HASH_WAIT_S", 0.05)
    saturated, real = threading.Semaphore(0), auth._hash_slots
    with TestClient(server.app) as other:
        monkeypatch.setattr(auth, "_hash_slots", saturated)
        assert other.post("/api/account/login", json=CREDS).status_code == 503
        monkeypatch.setattr(auth, "_hash_slots", real)  # the slot was never spent
        assert other.post("/api/account/login", json=CREDS).status_code == 200


def test_rate_limiter_key_space_is_bounded():
    limiter = auth.RateLimiter(5, window_s=3600)
    monkey = limiter.MAX_KEYS + 5000
    for i in range(monkey):
        limiter.consume(f"ip-{i}", now=float(i))
    assert len(limiter._hits) <= limiter.MAX_KEYS


def test_rate_limiter_release_gives_back_one_slot():
    limiter = auth.RateLimiter(1, window_s=60)
    limiter.consume("ip", now=0)
    with pytest.raises(auth.RateLimited):
        limiter.consume("ip", now=0)
    limiter.release("ip")
    limiter.consume("ip", now=0)  # usable again
    limiter.release("nobody")  # releasing an untracked key is harmless


def test_hasher_queue_gate_sheds_without_occupying_a_thread(client, monkeypatch):
    """Bounding hash *concurrency* bounds memory, but callers merely waiting
    still hold threads the trial flow needs — so past the queue bound we
    refuse immediately rather than waiting our turn."""
    monkeypatch.setattr(auth, "HASH_WAIT_S", 30)  # a wait here would be a 30s stall
    monkeypatch.setattr(auth, "_hash_queue", threading.Semaphore(0))
    started = time.monotonic()
    assert client.post("/api/account/signup", json=CREDS).status_code == 503
    assert time.monotonic() - started < 5


def test_a_crash_does_not_burn_a_signup_slot(client, monkeypatch):
    """argon2 raises HashingError when it can't get its 64 MiB — the very
    pressure the hasher caps exist for. That must not also cost the creation
    slots, or an overloaded box becomes an hour-long lockout."""
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(2, 3600))
    real_hash = auth.hash_password

    def boom(_):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(auth, "hash_password", boom)
    with TestClient(server.app, raise_server_exceptions=False) as c:
        for _ in range(3):
            assert c.post("/api/account/signup", json=CREDS).status_code == 500
    monkeypatch.setattr(auth, "hash_password", real_hash)
    assert client.post("/api/account/signup", json=CREDS).status_code == 200


def test_a_shed_signup_does_not_spend_an_attempt(client, monkeypatch):
    monkeypatch.setattr(server, "signup_attempt_limiter", auth.RateLimiter(2, 3600))
    monkeypatch.setattr(auth, "HASH_WAIT_S", 0.05)
    saturated, real = threading.Semaphore(0), auth._hash_slots
    monkeypatch.setattr(auth, "_hash_slots", saturated)
    for _ in range(3):
        assert client.post("/api/account/signup", json=CREDS).status_code == 503
    monkeypatch.setattr(auth, "_hash_slots", real)
    assert client.post("/api/account/signup", json=CREDS).status_code == 200


def test_successes_do_not_refund_other_requests_wrong_guesses(client, monkeypatch):
    """A refund gives back the slot that request took, not the last one taken,
    so someone holding one working account can't alternate correct/wrong to
    keep guessing at another. Wrong guesses still accumulate to the limit."""
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(3, 900))
    bad = {**CREDS, "password": "wrongwrongwrong"}
    codes = []
    with TestClient(server.app) as other:
        for _ in range(5):
            codes.append(other.post("/api/account/login", json=bad).status_code)
            codes.append(other.post("/api/account/login", json=CREDS).status_code)
    assert codes.count(400) == 3, codes  # the wrong guesses ran out of slots
    assert 429 in codes
