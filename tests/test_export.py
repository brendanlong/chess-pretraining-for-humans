"""Downloading your own record: what it contains, and what it can't reach."""

import csv
import io
import json

from fastapi.testclient import TestClient

from trainer import export, server

from .conftest import answer, answer_body, next_trial

CREDS = {"username": "exporter", "password": "correct horse", "email": "a@b.co"}
GAME_URL = "https://lichess.org/abcd1234#41"


def fetch(client, fmt="json"):
    r = client.get(f"/api/account/export?format={fmt}")
    assert r.status_code == 200, r.text
    return r


def exported(client):
    return json.loads(fetch(client).text)


def test_export_carries_the_account_and_every_answer(client, db):
    with db:
        db.execute("UPDATE items SET game_url = ?", (GAME_URL,))
    trial = next_trial(client)
    result = answer(client, trial)
    client.post("/api/account/signup", json=CREDS)
    stored = db.execute("SELECT rating, created_at FROM users").fetchone()

    data = exported(client)

    assert data["account"] == {
        "username": CREDS["username"],
        "guest": False,
        "email": CREDS["email"],
        "rating": round(stored["rating"], 1),
        "calibrating": result["calibrating"],
        "answers": 1,
        "created_at_utc": stored["created_at"],
    }
    (row,) = data["responses"]
    assert row["item_id"] == trial["item_id"]
    assert row["fen"] == trial["fen"]
    assert row["choice_uci"] == trial["moves"][0]["uci"]
    assert row["choice_san"] == trial["moves"][0]["san"]
    assert row["correct"] == result["correct"]
    assert row["best_uci"] == result["best"]["uci"]
    assert row["best_san"] == result["best"]["san"]
    assert row["distractor_uci"] == result["distractor"]["uci"]
    assert row["game_url"] == GAME_URL
    assert row["from_share_link"] is False
    assert row["answered_at_utc"]
    assert row["rating_after"] == round(stored["rating"], 1)


def test_a_guest_can_export_before_making_an_account(client):
    answer(client, next_trial(client))

    data = exported(client)

    assert data["account"]["guest"] is True
    assert data["account"]["username"] is None
    assert data["account"]["email"] is None
    assert len(data["responses"]) == 1


def test_timing_is_kept_as_it_was_recorded(client):
    trial = next_trial(client)
    body = {**answer_body(trial), "response_ms": 1234}
    assert client.post("/api/answer", json=body).status_code == 200

    (row,) = exported(client)["responses"]

    assert row["response_ms"] == 1234


def test_an_answer_from_a_share_link_says_so(client):
    answer(client, next_trial(client))  # answering is what mints the identity
    linked = client.get(f"/api/next?item={next_trial(client)['item_id']}").json()
    answer(client, linked)

    rows = exported(client)["responses"]

    assert [r["from_share_link"] for r in rows] == [False, True]


def test_export_never_reaches_an_item_this_user_has_not_answered(client, db):
    trial = next_trial(client)
    answer(client, trial)

    rows = exported(client)["responses"]

    assert db.execute("SELECT COUNT(*) FROM items").fetchone()[0] > 1  # one to be missing
    assert [r["item_id"] for r in rows] == [trial["item_id"]]


def test_export_holds_one_session_s_answers_and_not_another_s(client):
    answer(client, next_trial(client))
    with TestClient(server.app) as other:
        answer(other, next_trial(other))
        answer(other, next_trial(other))

        assert len(exported(other)["responses"]) == 2
    assert len(exported(client)["responses"]) == 1


def test_a_caller_with_no_session_has_nothing_to_export(client):
    r = client.get("/api/account/export")

    assert r.status_code == 400
    assert "nothing here to export" in r.json()["detail"]
    # Refusing must not be what mints a row.
    assert client.get("/api/stats").json()["attempts"] == 0


def test_csv_is_one_row_per_answer_under_the_declared_columns(client):
    answer(client, next_trial(client))
    answer(client, next_trial(client))

    r = fetch(client, "csv")

    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 2
    assert list(rows[0]) == export.RESPONSE_COLUMNS
    assert rows[0]["fen"] and rows[0]["choice_san"]


def test_both_formats_arrive_as_named_attachments(client):
    answer(client, next_trial(client))

    for fmt in export.FORMATS:
        disposition = fetch(client, fmt).headers["content-disposition"]
        assert disposition.startswith("attachment; ")
        assert disposition.endswith(f'.{fmt}"')


def test_export_is_never_cached(client):
    answer(client, next_trial(client))

    assert fetch(client).headers["cache-control"] == "no-store"


def test_an_unknown_format_is_refused_rather_than_guessed_at(client):
    answer(client, next_trial(client))

    r = client.get("/api/account/export?format=xml")

    assert r.status_code == 400
    assert "json" in r.json()["detail"]


def test_deleting_the_account_erases_exactly_what_was_exported(client):
    answer(client, next_trial(client))
    answer(client, next_trial(client))
    client.post("/api/account/signup", json=CREDS)
    exported_rows = len(exported(client)["responses"])

    r = client.post("/api/account/delete", json={"password": CREDS["password"]})

    assert r.json()["responses_deleted"] == exported_rows
