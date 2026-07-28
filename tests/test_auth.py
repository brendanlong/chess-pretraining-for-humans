import pytest
from fastapi.testclient import TestClient

from trainer import auth, server

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


def test_login_rate_limit_is_per_ip(client, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    monkeypatch.setattr(server, "login_limiter", auth.RateLimiter(2, 900))
    with TestClient(server.app) as other:
        bad = {**CREDS, "password": "wrongwrongwrong"}
        assert other.post("/api/account/login", json=bad).status_code == 400
        assert other.post("/api/account/login", json=bad).status_code == 400
        # Right password, but out of tries.
        assert other.post("/api/account/login", json=CREDS).status_code == 429


def test_signup_rate_limit_is_per_ip(db, monkeypatch):
    monkeypatch.setattr(server, "signup_limiter", auth.RateLimiter(1, 3600))
    with TestClient(server.app) as a, TestClient(server.app) as b:
        assert a.post("/api/account/signup", json=CREDS).status_code == 200
        assert b.post("/api/account/signup", json={**CREDS, "username": "sec"}).status_code == 429


def test_rate_limiter_window_expires():
    limiter = auth.RateLimiter(2, window_s=60)
    limiter.check("ip", now=0)
    limiter.check("ip", now=1)
    with pytest.raises(auth.RateLimited):
        limiter.check("ip", now=2)
    limiter.check("ip", now=100)  # the first two have rolled out of the window


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
