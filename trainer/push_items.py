"""Ship a freshly labeled item bank into a deployment without touching responses.

Mining and labeling stay local — they need Stockfish, zstd and hours of CPU —
so a refreshed bank has to cross a machine boundary. It must never cross as a
file copy: `responses` is the experimental record and lives in the same SQLite
file, so overwriting the live database destroys it, and restoring from a backup
to undo that loses everything answered since.

Instead the bank travels as its own SQLite file (`export`) and is merged into
the live one row by row (`merge`). Items are matched on `fen`, not on `id`:
ids are assigned per-database and `responses.item_id` points at the live ones.

    local$  uv run python -m trainer.push_items export --out /tmp/items.db
    server$ python -m trainer.push_items merge /tmp/items.db

Positions already in the bank are skipped rather than updated. An item whose
labels changed underneath the answers already given to it would make those
responses uninterpretable — the recorded `correct` was decided against the old
best move. Relabelling in place is a different, rarer operation and deliberately
isn't this tool.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from .db import DEFAULT_DB, connect

# `id` is per-database; `attempts`/`correct` are the tally of answers given on
# whichever deployment the row lives on, and those answers aren't coming along.
PER_DATABASE_COLUMNS = {"id", "attempts", "correct"}


def item_columns(conn: sqlite3.Connection, schema: str = "main") -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA {schema}.table_info(items)")]


def shared_columns(conn: sqlite3.Connection, a: str, b: str) -> list[str]:
    """Columns `items` has in both schemas, minus the per-database ones.

    The two databases are usually the same schema version, but the whole point
    of this tool is that they're upgraded at different times.
    """
    other = set(item_columns(conn, b))
    cols = [c for c in item_columns(conn, a) if c in other and c not in PER_DATABASE_COLUMNS]
    if not cols:
        sys.exit(f"no shared 'items' columns between {a} and {b} — is the source an item bank?")
    return cols


def export(db: Path, out: Path) -> int:
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    src = connect(db)
    connect(out).close()  # same schema, same migrations
    src.execute("ATTACH DATABASE ? AS dest", (str(out),))
    try:
        cols = ", ".join(shared_columns(src, "main", "dest"))
        n = src.execute(f"INSERT INTO dest.items ({cols}) SELECT {cols} FROM main.items").rowcount
        src.commit()
    finally:
        src.execute("DETACH DATABASE dest")
        src.close()
    return n


def merge(db: Path, incoming: Path, dry_run: bool = False) -> tuple[int, int]:
    """Insert every position the live bank doesn't have yet. Returns (added, skipped)."""
    if not incoming.exists():
        sys.exit(f"{incoming} does not exist")
    conn = connect(db)
    conn.execute("ATTACH DATABASE ? AS inc", (str(incoming),))
    try:
        cols = ", ".join(shared_columns(conn, "inc", "main"))
        offered = conn.execute("SELECT COUNT(*) FROM inc.items").fetchone()[0]
        # `WHERE true` is what lets SQLite tell the upsert clause apart from a
        # join condition in INSERT … SELECT.
        added = conn.execute(
            f"INSERT INTO main.items ({cols}) SELECT {cols} FROM inc.items"
            " WHERE true ON CONFLICT(fen) DO NOTHING"
        ).rowcount
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        conn.execute("DETACH DATABASE inc")
        conn.close()
    return added, offered - added


def main() -> None:
    ap = argparse.ArgumentParser(description="Move a labeled item bank between databases.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="the bank to read/write")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="copy the items table into a fresh, standalone file")
    e.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge", help="add an exported file's unseen positions to --db")
    m.add_argument("incoming", type=Path)
    m.add_argument("--dry-run", action="store_true", help="report the counts, write nothing")
    args = ap.parse_args()

    if args.cmd == "export":
        print(f"exported {export(args.db, args.out)} items to {args.out}")
    else:
        added, skipped = merge(args.db, args.incoming, args.dry_run)
        print(
            f"{'would add' if args.dry_run else 'added'} {added} items"
            f", skipped {skipped} already in the bank"
        )


if __name__ == "__main__":
    main()
