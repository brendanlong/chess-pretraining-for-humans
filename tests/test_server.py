from fastapi.testclient import TestClient

from trainer import auth, server

from .conftest import answer, next_trial


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


def test_separate_browsers_get_separate_identities(db):
    with TestClient(server.app) as a, TestClient(server.app) as b:
        answer(a, next_trial(a))
        assert a.get("/api/stats").json()["attempts"] == 1
        assert b.get("/api/stats").json()["attempts"] == 0
        assert next_trial(b)["repeat"] is False  # b's bank is untouched
