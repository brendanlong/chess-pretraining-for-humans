"""Everything one user's record holds, in a file they can keep.

Two jobs, and they pull in different directions, which is why there are two
formats rather than one. The first is to show what the app is holding: the
privacy policy describes it in prose, and a file that *is* it can't drift out
of date the way prose can. That has to be the whole account, so it's JSON. The
second is to be useful — a column per fact, openable in a spreadsheet — which
is CSV, and CSV is one table, so it is the answers alone.

The set exported here is deliberately the same set `auth.delete_user` erases,
so "download it" and "delete it" are the same promise read two ways. Withheld
are only the password hash — not the user's data to have back: it exists so
that nobody, us included, holds the password — and internal bookkeeping like
row ids, which say nothing about them.

An answer's own row means little on its own — an item id and a UCI move — so
each one is joined to the position it was about. That is only ever an item this
user has already answered and seen the reveal for, so it tells them nothing the
app hasn't; nothing here reaches an item they haven't.
"""

import csv
import io
import json
from datetime import UTC, datetime

import chess

from . import auth, db, rating

# Read in one pass over `idx_responses_user`, with a primary-key seek per item.
# Oldest first: this is a history, and a history reads forwards. An inner join
# drops nothing — `responses.item_id` is an enforced foreign key, so the item a
# response names is still there.
RESPONSES_SQL = """
    SELECT r.created_at, r.item_id, r.choice_uci, r.correct, r.response_ms,
           r.user_rating_before, r.user_rating_after, r.item_rating_before, r.shared,
           i.fen, i.best_uci, i.distractor_uci, i.game_url
      FROM responses r JOIN items i ON i.id = r.item_id
     WHERE r.user_id = ?
     ORDER BY r.id"""

# The CSV's column order, and the only place it is stated: the writer takes it
# from here, so a field added to a row without a column to put it in fails
# loudly instead of being dropped from the spreadsheet half of the export.
RESPONSE_COLUMNS = [
    # Every timestamp here carries `_utc` in its name. The database stores them
    # without a zone, and a bare "2026-08-01 21:30:08" in a file somebody opens
    # months later is a number they have to come back and ask about. Saying it
    # in the column name rather than in the value keeps the value a date a
    # spreadsheet will parse.
    "answered_at_utc",
    "item_id",
    "fen",
    "choice_uci",
    "choice_san",
    "correct",
    "response_ms",
    "rating_before",
    "rating_after",
    "item_rating",
    "from_share_link",
    "best_uci",
    "best_san",
    "distractor_uci",
    "distractor_san",
    "game_url",
]


def _rating(value: float | None) -> float | None:
    """Ratings are carried at full precision and shown rounded; one decimal is
    enough to reconstruct any move the app made without implying more."""
    return None if value is None else round(value, 1)


def _san(board: chess.Board, uci: str) -> str:
    return board.san(chess.Move.from_uci(uci))


def response_rows(conn: db.Queryable, user_id: int) -> list[dict]:
    """Every answer this user has given, with the position it was about.

    One `chess.Board` per row serves all three moves: parsing a FEN is the
    expensive part, and a long history is the case this has to survive.
    """
    rows = []
    for r in conn.execute(RESPONSES_SQL, (user_id,)):
        board = chess.Board(r["fen"])
        rows.append(
            {
                "answered_at_utc": r["created_at"],
                "item_id": r["item_id"],
                "fen": r["fen"],
                "choice_uci": r["choice_uci"],
                "choice_san": _san(board, r["choice_uci"]),
                "correct": bool(r["correct"]),
                "response_ms": r["response_ms"],
                "rating_before": _rating(r["user_rating_before"]),
                "rating_after": _rating(r["user_rating_after"]),
                "item_rating": _rating(r["item_rating_before"]),
                "from_share_link": bool(r["shared"]),
                "best_uci": r["best_uci"],
                "best_san": _san(board, r["best_uci"]),
                "distractor_uci": r["distractor_uci"],
                "distractor_san": _san(board, r["distractor_uci"]),
                "game_url": r["game_url"],
            }
        )
    return rows


def account_row(user: dict) -> dict:
    """The `users` row, as the person it belongs to would read it.

    Three of its columns aren't here. The password hash never leaves the
    database. The internal id names this row only inside a database nobody
    else can reach a row of, so handing it over says nothing about the user
    and something about how many of them there are. And the raw calibration
    step is a staircase's internal state — `calibrating` is the fact it
    encodes, which is also what the app shows.
    """
    return {
        "username": auth.display_name(user),  # null while this is still a guest
        "guest": auth.is_guest(user),
        "email": user["email"],
        "rating": _rating(user["rating"]),
        "calibrating": rating.is_calibrating(user["calib_step"]),
        "answers": user["attempts"],
        "created_at_utc": user["created_at"],
    }


def as_json(user: dict, responses: list[dict], now: datetime) -> bytes:
    return json.dumps(
        {
            "exported_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "account": account_row(user),
            "responses": responses,
        },
        indent=2,
    ).encode()


def as_csv(responses: list[dict]) -> bytes:
    # `newline=""` and the writer's own CRLF, which is what RFC 4180 asks for
    # and what the spreadsheets this is aimed at expect. Every value is ASCII —
    # timestamps, FENs, SAN, and a Lichess URL — so no byte-order mark is
    # needed to keep Excel from mangling it.
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, RESPONSE_COLUMNS)
    writer.writeheader()
    writer.writerows(responses)
    return out.getvalue().encode()


def filename(fmt: str, now: datetime) -> str:
    """Dated, and named after the app rather than the user: a username would
    have to be quoted into a header, and the file is already in their hands."""
    return f"chess-pretraining-{now:%Y-%m-%d}.{fmt}"


# What the endpoint will accept, named here so the two can't disagree.
FORMATS = ("json", "csv")


def build(conn: db.Queryable, user: dict, fmt: str) -> tuple[bytes, str, str]:
    """The file to send: (body, media type, filename). `fmt` is one of FORMATS."""
    now = datetime.now(UTC)
    responses = response_rows(conn, user["id"])
    if fmt == "csv":
        return as_csv(responses), "text/csv; charset=utf-8", filename("csv", now)
    return as_json(user, responses, now), "application/json", filename("json", now)
