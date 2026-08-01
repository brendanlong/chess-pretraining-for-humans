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

Positions already in the bank are skipped rather than updated, with one
exception named on `MEASURED_COLUMNS`. An item whose labels changed underneath
the answers already given to it would make those responses uninterpretable —
the recorded `correct` was decided against the old best move. Relabelling in
place is a different, rarer operation and deliberately isn't this tool.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from .db import DEFAULT_DB, connect

# `id` is per-database; everything else about an item is a property of the
# position, so it travels.
PER_DATABASE_COLUMNS = {"id"}
# What `merge` will fill in on a position the live bank already holds: the
# lookahead ladder, the two readings taken off it, and the difficulty that
# follows. All of them or none — `shallow_gap` is a reading of `gap_ladder`,
# `learnable` is a reading of `solution_depth`, and `rating` is a function of
# `shallow_gap` — so landing one without the others would leave the bank saying
# several different things about the same item.
#
# `rating` is recomputed here rather than left to `db.connect`'s re-derivation
# because the merge runs against a database a *server* has open, and that server
# ran its migrations at boot. Nothing would put the new difficulty in front of a
# user until the machine restarted, which is not a step this runbook has.
MEASURED_COLUMNS = ("solution_depth", "gap_ladder", "shallow_gap", "learnable")
# A row nobody has run the ladder over. Just the NULL: 0 is the ladder saying no
# depth settles it, which is a verdict and not an absence. `learnable` gets no
# say — on a live row it is the *old* single-depth check's word, taken on a hash
# the deep pass had just filled, so a row carrying 0 there has still never been
# measured and `trainer.backfill_depth` would rightly re-measure it.
_UNMEASURED = "{table}.solution_depth IS NULL"


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


def detach(conn: sqlite3.Connection, schema: str) -> None:
    """Detach, rolling back first.

    SQLite refuses to detach inside a transaction, and a failed insert leaves
    one open — so without the rollback the cleanup raises a lock error on the
    way out and buries the failure that actually happened. After a commit the
    rollback is a no-op.
    """
    conn.rollback()
    conn.execute(f"DETACH DATABASE {schema}")


def export(db: Path, out: Path) -> int:
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    src = connect(db)
    connect(out).close()  # same schema, same migrations
    src.execute("ATTACH DATABASE ? AS dest", (str(out),))
    done = False
    try:
        cols = ", ".join(shared_columns(src, "main", "dest"))
        n = src.execute(f"INSERT INTO dest.items ({cols}) SELECT {cols} FROM main.items").rowcount
        src.commit()
        done = True
    finally:
        detach(src, "dest")
        src.close()
        if not done:
            # Leaving the half-written file behind would be worse than useless:
            # the next run refuses to overwrite it, so the retry fails too. The
            # sidecars have to go with it, or the retry opens a fresh database
            # next to a stale WAL. Removing them after the close, so there's
            # nothing left to write them back.
            for path in (out, Path(f"{out}-wal"), Path(f"{out}-shm")):
                path.unlink(missing_ok=True)
    return n


def merge(db: Path, incoming: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Insert every position the live bank doesn't have yet.

    Returns (added, skipped, measured) — the last being positions already held
    whose lookahead depth the incoming bank knows and the live one doesn't.
    """
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
        # The one thing a held position does take from the incoming bank. It is
        # not the relabelling this tool refuses: lookahead depth needs Stockfish
        # and so can only be measured on the pipeline's side of the boundary,
        # and filling in a blank changes how hard the item is *said* to be, and
        # whether it is noise, without touching which move is correct — so the
        # answers already given to it still mean what they meant. Only blanks:
        # a row that has been measured keeps its verdict, since re-measuring one
        # is relabelling.
        #
        # Skipped outright when either side predates the column. Schema skew is
        # this tool's whole premise, and a bank exported from an older checkout
        # has to still deliver its items rather than raise on the way past — the
        # failure would roll the inserts back with it.
        measured = 0
        both = set(item_columns(conn, "inc")) & set(item_columns(conn, "main"))
        if set(MEASURED_COLUMNS) <= both:
            fill = ", ".join(MEASURED_COLUMNS)
            measured = conn.execute(
                f"UPDATE main.items SET ({fill}, rating) = (SELECT {fill},"
                f"    difficulty_rating(inc.items.shallow_gap)"
                f"    FROM inc.items WHERE inc.items.fen = main.items.fen)"
                f" WHERE {_UNMEASURED.format(table='main.items')}"
                f"   AND EXISTS (SELECT 1 FROM inc.items WHERE inc.items.fen = main.items.fen"
                f"               AND NOT ({_UNMEASURED.format(table='inc.items')}))"
            ).rowcount
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        detach(conn, "inc")
        conn.close()
    return added, offered - added, measured


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
        added, skipped, measured = merge(args.db, args.incoming, args.dry_run)
        print(
            f"{'would add' if args.dry_run else 'added'} {added} items"
            f", skipped {skipped} already in the bank"
            f", filled in lookahead depth on {measured} of them"
        )


if __name__ == "__main__":
    main()
