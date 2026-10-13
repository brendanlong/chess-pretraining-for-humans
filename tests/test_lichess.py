import json
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from trainer import account, lichess, server

from .conftest import answer, next_trial

CREDS = {"username": "tester", "password": "correct horse battery", "email": None}
PERFS = {"blitz": {"games": 120, "rating": 1650, "rd": 60, "prog": 12}}


class FakeLichess:
    """lichess.org, as far as the flow can tell: a code names an account."""

    def __init__(self, monkeypatch):
        self.accounts = {"c-alice": {"id": "alice", "username": "Alice", "perfs": PERFS}}
        self.exchanged = []
        self.revoked = []
        self.down = False
        monkeypatch.setattr(lichess, "exchange", self.exchange)
        monkeypatch.setattr(lichess, "account", self.account)
        monkeypatch.setattr(lichess, "revoke", self.revoke)
        monkeypatch.setattr(lichess, "users", self.users)

    def exchange(self, code, verifier, redirect_uri):
        if self.down:
            raise lichess.LichessError("unreachable")
        self.exchanged.append((code, verifier, redirect_uri))
        return f"token-{code}"

    def account(self, token):
        return self.accounts[token.removeprefix("token-")]

    def revoke(self, token):
        self.revoked.append(token)

    def users(self, ids):
        by_id = {a["id"]: a for a in self.accounts.values()}
        return [by_id[i] for i in ids if i in by_id]


@pytest.fixture
def fake(monkeypatch):
    return FakeLichess(monkeypatch)


def start(client) -> str:
    r = client.get("/api/lichess/login", follow_redirects=False)
    assert r.status_code == 303
    return parse_qs(urlparse(r.headers["location"]).query)["state"][0]


def come_back(client, **params) -> str:
    r = client.get("/api/lichess/callback", params=params, follow_redirects=False)
    assert r.status_code == 303
    assert "lichess_oauth=" in r.headers["set-cookie"]  # spent either way
    return parse_qs(urlparse(r.headers["location"]).query)["lichess"][0]


def sign_in(client, code="c-alice") -> str:
    return come_back(client, code=code, state=start(client))


def snapshots(db) -> list:
    return db.execute(
        "SELECT user_id, lichess_id, perfs FROM lichess_ratings ORDER BY id"
    ).fetchall()


def test_login_sends_the_browser_to_lichess_with_pkce(client):
    r = client.get("/api/lichess/login", follow_redirects=False)
    target = urlparse(r.headers["location"])
    q = {k: v[0] for k, v in parse_qs(target.query).items()}
    assert f"{target.scheme}://{target.netloc}{target.path}" == f"{lichess.HOST}/oauth"
    assert q["response_type"] == "code"
    assert q["code_challenge_method"] == "S256"
    assert q["redirect_uri"] == "http://testserver/api/lichess/callback"
    assert "scope" not in q
    state, verifier = client.cookies["lichess_oauth"].split(".")
    assert q["state"] == state
    assert q["code_challenge"] == lichess.challenge(verifier)
    assert verifier not in r.headers["location"]
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Path=/api/lichess/" in cookie and "SameSite=lax" in cookie


def test_the_exchange_carries_the_verifier_and_the_same_redirect(client, fake):
    state = start(client)
    verifier = client.cookies["lichess_oauth"].split(".")[1]
    come_back(client, code="c-alice", state=state)
    assert fake.exchanged == [("c-alice", verifier, "http://testserver/api/lichess/callback")]
    assert fake.revoked == ["token-c-alice"]


def test_a_callback_this_browser_did_not_start_is_refused(client, db, fake):
    """The state check is the CSRF defence: a code for someone else's account
    must not be linked to whoever happens to follow the link."""
    assert come_back(client, code="c-alice", state="forged") == "expired"
    start(client)
    assert come_back(client, code="c-alice", state="wrong") == "expired"
    assert fake.exchanged == []
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_cancelling_on_lichess_says_so(client, db, fake):
    assert come_back(client, error="access_denied", state=start(client)) == "denied"
    assert fake.exchanged == []


def test_lichess_failing_writes_nothing(client, db, fake):
    fake.down = True
    assert sign_in(client) == "failed"
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_signing_in_fresh_creates_an_account_with_a_snapshot(client, db, fake):
    assert sign_in(client) == "created"
    assert client.get("/api/account").json() == {
        "username": "Alice",
        "guest": False,
        "password": False,
        "lichess": "Alice",
    }
    ((user_id, lichess_id, perfs),) = snapshots(db)
    assert lichess_id == "alice" and json.loads(perfs) == PERFS
    # The Lichess name is shown, not taken: the row's own name stays opaque,
    # so a password signup can still have "Alice".
    assert db.execute("SELECT name FROM users WHERE id = ?", (user_id,)).fetchone()[0] != "Alice"


def test_a_guest_linking_keeps_its_history_on_a_rotated_session(client, db, fake):
    answer(client, next_trial(client))
    before = client.cookies[server.auth.COOKIE_NAME]

    assert sign_in(client) == "linked"

    assert client.cookies[server.auth.COOKIE_NAME] != before
    assert client.get("/api/stats").json()["attempts"] == 1
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_signing_in_again_elsewhere_reaches_the_same_row(client, db, fake):
    answer(client, next_trial(client))
    sign_in(client)
    with TestClient(server.app) as other:
        assert sign_in(other) == "signed-in"
        assert other.get("/api/stats").json()["attempts"] == 1
    assert len(snapshots(db)) == 2  # one per sign-in


def link(client, password=CREDS["password"]) -> str:
    """The password-checked way into the flow, for an account with one."""
    r = client.post("/api/lichess/link", json={"password": password})
    assert r.status_code == 200, r.text
    return parse_qs(urlparse(r.json()["url"]).query)["state"][0]


def test_a_password_account_links_with_its_password_and_can_unlink(client, db, fake):
    client.post("/api/account/signup", json=CREDS)
    assert come_back(client, code="c-alice", state=link(client)) == "linked"
    account = client.get("/api/account").json()
    assert account["username"] == "tester" and account["lichess"] == "Alice"

    assert client.post("/api/lichess/unlink", json={"password": "nope"}).status_code == 400
    r = client.post("/api/lichess/unlink", json={"password": CREDS["password"]})
    assert r.status_code == 200, r.text
    assert r.json()["lichess"] is None
    assert snapshots(db) == []
    assert tuple(db.execute("SELECT lichess_id, lichess_linked_at FROM users").fetchone()) == (
        None,
        None,
    )


def test_linking_a_password_account_without_the_password_is_refused(client, db, fake):
    """Otherwise a stolen or unattended session could link the thief's own
    Lichess account and keep a way in after the session is gone."""
    client.post("/api/account/signup", json=CREDS)
    assert client.post("/api/lichess/link", json={"password": "nope"}).status_code == 400
    assert sign_in(client) == "password"
    assert db.execute("SELECT lichess_id FROM users").fetchone()[0] is None
    assert snapshots(db) == []


def test_a_link_grant_is_for_the_account_that_earned_it(client, db, fake):
    client.post("/api/account/signup", json=CREDS)
    state = link(client)
    client.post("/api/account/logout", json={})
    client.post("/api/account/signup", json={**CREDS, "username": "other"})
    assert come_back(client, code="c-alice", state=state) == "password"


def test_unlinking_takes_a_json_body(client, fake):
    client.post("/api/account/signup", json=CREDS)
    come_back(client, code="c-alice", state=link(client))
    r = client.post("/api/lichess/unlink", data={"password": CREDS["password"]})
    assert r.status_code == 422


def test_set_password_unlinks_unless_told_otherwise(client, db, fake, monkeypatch):
    client.post("/api/account/signup", json=CREDS)
    come_back(client, code="c-alice", state=link(client))
    monkeypatch.setattr(account.getpass, "getpass", lambda _prompt: "a new password")
    user = server.auth.find_by_username(db, "tester")
    assert user is not None
    assert account.set_password(db, user, None, keep_lichess=True) == 0
    assert db.execute("SELECT lichess_id FROM users").fetchone()[0] == "alice"
    assert account.set_password(db, user, None) == 0
    assert db.execute("SELECT lichess_id FROM users").fetchone()[0] is None
    assert snapshots(db) == []


def test_an_account_made_through_lichess_cannot_unlink(client, db, fake):
    sign_in(client)
    r = client.post("/api/lichess/unlink", json={})
    assert r.status_code == 400
    assert len(snapshots(db)) == 1


def test_a_second_lichess_account_is_its_own_row(client, db, fake):
    """Re-pointing a linked row's link would swap whose ratings sit beside
    its answers, so another Lichess account gets a row of its own."""
    fake.accounts["c-bob"] = {"id": "bob", "username": "bob", "perfs": {}}
    sign_in(client)
    assert sign_in(client, "c-bob") == "created"
    assert client.get("/api/account").json()["username"] == "bob"
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2


def test_linking_moves_nothing_the_trial_flow_reads(client, db, fake):
    fake.accounts["c-alice"]["perfs"] = {"blitz": {"rating": 2800, "games": 5000}}
    answer(client, next_trial(client))
    before = db.execute("SELECT rating, calib_step, attempts FROM users").fetchone()
    sign_in(client)
    assert db.execute("SELECT rating, calib_step, attempts FROM users").fetchone() == before
    assert next_trial(client)["calibrating"]


def test_deleting_a_lichess_account_needs_no_password_and_takes_the_snapshots(client, db, fake):
    sign_in(client)
    answer(client, next_trial(client))
    r = client.post("/api/account/delete", json={"password": ""})
    assert r.status_code == 200, r.text
    assert snapshots(db) == []
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_export_carries_the_link_and_the_snapshots(client, fake):
    sign_in(client)
    answer(client, next_trial(client))
    data = client.get("/api/account/export?format=json").json()
    assert data["account"]["lichess_username"] == "Alice"
    assert data["account"]["lichess_linked_at_utc"]
    (snap,) = data["lichess_ratings"]
    assert snap["perfs"] == PERFS and snap["lichess_id"] == "alice"


def test_refresh_snapshots_only_who_is_due(client, db, fake):
    fake.accounts["c-bob"] = {"id": "bob", "username": "bob", "perfs": {}, "disabled": True}
    fake.accounts["c-carol"] = {"id": "carol", "username": "carol", "perfs": {}}
    for code in ("c-alice", "c-bob", "c-carol"):
        with TestClient(server.app) as browser:
            sign_in(browser, code)
    assert server.refresh_lichess_ratings() == 0  # everyone was just snapshotted

    with db:
        db.execute("UPDATE lichess_ratings SET taken_at = datetime('now', '-8 days')")
    # A closed account is skipped, not unlinked: it may reopen.
    assert server.refresh_lichess_ratings() == 2
    assert server.refresh_lichess_ratings() == 0
    counts = dict(
        db.execute("SELECT lichess_id, COUNT(*) FROM lichess_ratings GROUP BY lichess_id")
    )
    assert counts == {"alice": 2, "bob": 1, "carol": 2}
