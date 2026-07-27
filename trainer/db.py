"""SQLite storage for items, users, and responses."""

import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data/items.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    fen TEXT NOT NULL UNIQUE,
    best_uci TEXT NOT NULL,
    distractor_uci TEXT NOT NULL,
    distractor_source TEXT NOT NULL,  -- 'game' (move actually played) or 'multipv'
    cp_best INTEGER,                  -- mover POV, deep search
    mate_best INTEGER,
    cp_distractor INTEGER,
    mate_distractor INTEGER,
    wp_best REAL NOT NULL,
    wp_distractor REAL NOT NULL,
    gap_wp REAL NOT NULL,
    learnable INTEGER NOT NULL,       -- shallow search agrees which move is better
    depth_deep INTEGER NOT NULL,
    depth_shallow INTEGER NOT NULL,
    rating REAL NOT NULL,             -- adaptive difficulty rating, seeded from gap_wp
    attempts INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0,
    ply INTEGER,
    game_url TEXT,
    mover_elo INTEGER,
    time_control TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    rating REAL NOT NULL DEFAULT 1500,
    attempts INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    choice_uci TEXT NOT NULL,
    correct INTEGER NOT NULL,
    probe INTEGER NOT NULL DEFAULT 0,  -- no-feedback trial
    response_ms INTEGER,
    user_rating_before REAL,
    user_rating_after REAL,
    item_rating_before REAL,
    item_rating_after REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_items_rating ON items(rating);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses(user_id, id);
"""


def connect(path: Path = DEFAULT_DB, check_same_thread: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn
