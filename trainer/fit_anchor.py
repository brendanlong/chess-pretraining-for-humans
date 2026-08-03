"""Measure the offset between the user and item scales from live answers.

`rating.RESPONSE_ANCHOR` is the number this fits. Every scored response
stores the two ratings it was scored against, so the record can be asked
directly: given what the model believed at each moment, how far off was it?
The answer comes back in rating points — the uniform shift that makes the
model's expected scores match the accuracy users actually produced.

    uv run python -m trainer.fit_anchor --db copy-of-live.db   # the anchor
    uv run python -m trainer.fit_anchor --link                 # the link shape

What it can and can't say. The user ratings in the record are the model's own
running estimates, learned against the item ratings being measured — so this
is a consistency check at the operating point, not an independent scale. A
*uniform* offset is exactly what it estimates well: selection holds every
user at the same expected score, so a systematic gap between expected and
actual is the model disagreeing with everybody at once. What it cannot do is
re-shape the difficulty curve — an item's residual is confounded with who was
sent to it. That is issue #27's IRT model, which estimates abilities and
difficulties jointly; this is the cheap estimator that says whether the scale
is off and by how much, and `--link` is the first look at whether the Elo
logistic (400-point base, chance floor) is even the right family.

Fits are reported per scale era: `meta.anchored_at` (and `regraded_at` before
it) mark the moments stored snapshots changed scale, and pooling across a
boundary would average two different mistakes. On a healthy record the era
under the current constants fits an anchor near zero.

numpy is a dev-group dependency and not in the deployment; nothing the server
does imports this. Run against a *copy* of the live file — the module opens
it without migrations, but a copy is what makes that a guarantee rather than
a habit.
"""

import argparse
import itertools
import math
import sqlite3
from pathlib import Path

import numpy as np

from .db import DEFAULT_DB, open_connection
from .rating import _TARGET_OFFSET, K_USER, TARGET_ACCURACY

# The plateau `expected_score`'s chance floor puts in the likelihood is why
# this is a bracketed scan rather than a derivative method. The range is the
# assertion that a miss bigger than this is not a scale offset but a broken
# model, which deserves a traceback's worth of attention rather than a number.
ANCHOR_RANGE = (-800.0, 800.0)
MIN_BAND = 150  # a per-band anchor over fewer scored answers swings too hard to read


def _expected(u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """`rating.expected_score`, vectorised — one formula, stated twice, held
    together by `test_fit_anchor`."""
    return np.maximum(0.5, 1.0 / (1.0 + 10.0 ** ((i - u) / 400.0)))


def load(conn: sqlite3.Connection) -> np.ndarray:
    """Every scored answer, as a structured array, oldest first.

    Scored means the rating moved: repeats deliberately move nothing, and a
    row clamped still at a rating bound was scored against a model the bound
    was already overruling. Both are dropped by the same comparison.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(responses)")}
    calib = "calibrating" if "calibrating" in cols else "NULL AS calibrating"
    rows = conn.execute(
        f"SELECT id, user_id, correct, user_rating_before, item_rating_before,"
        f" user_rating_after - user_rating_before AS delta,"
        f" response_ms, shared, {calib}, created_at FROM responses"
        " WHERE user_rating_before IS NOT NULL AND item_rating_before IS NOT NULL"
        "   AND user_rating_before != user_rating_after ORDER BY id"
    ).fetchall()
    data = np.array(
        [
            (
                r["id"],
                r["user_id"],
                r["correct"],
                r["user_rating_before"],
                r["item_rating_before"],
                r["delta"],
                -1 if r["response_ms"] is None else r["response_ms"],
                r["shared"],
                -1 if r["calibrating"] is None else r["calibrating"],
                str(r["created_at"]),
            )
            for r in rows
        ],
        dtype=[
            ("id", "i8"),
            ("user", "i8"),
            ("correct", "i8"),
            ("u", "f8"),
            ("i", "f8"),
            ("delta", "f8"),
            ("ms", "i8"),
            ("shared", "i8"),
            ("calibrating", "i8"),
            ("at", "U19"),
        ],
    )
    return data


def elo_scored(data: np.ndarray) -> np.ndarray:
    """The rows Elo scored *after* the user's staircase had finished.

    Two reasons to drop a row, and one mask serves both. A staircase move
    follows rules the model under test had no part in. And any row while the
    staircase still owned the rating — including a shared answer Elo scored
    mid-calibration — carries a `user_rating_before` that is a transient, not
    an estimate, which would smear the fit. So: keep everything after each
    user's last staircase move. Recorded rows say which those were
    (`calibrating` and not `shared`); rows from before the column are inferred
    from the move size, which only the staircase (step >= CALIB_END_STEP) can
    push past K_USER — the inference the `calibrating` column exists to retire,
    since a clamp at a rating bound can shrink a staircase move under it.
    """
    recorded = data["calibrating"] >= 0
    staircase = np.where(
        recorded,
        (data["calibrating"] == 1) & (data["shared"] == 0),
        np.abs(data["delta"]) > K_USER,
    )
    keep = np.ones(len(data), bool)
    for user in np.unique(data["user"][staircase]):
        mine = data["user"] == user
        keep[mine & (data["id"] <= data["id"][mine & staircase].max())] = False
    return data[keep]


def log_likelihood(data: np.ndarray, anchor: float, scale: float = 400.0, floor2afc=False):
    """Mean log-likelihood of the answers under a shifted (and rescaled) link.

    `floor2afc` swaps the Elo-logistic-with-floor for the psychometric 2AFC
    form 0.5 + 0.5·logistic — the candidate the chance floor gestures at. The
    two families agree nowhere except near certainty, so comparing their fits
    is how the record gets to vote on the link and not just the location.
    """
    x = (data["u"] - (data["i"] + anchor)) / scale * 400.0
    if floor2afc:
        p = 0.5 + 0.5 / (1.0 + 10.0 ** (-x / 400.0))
    else:
        p = np.maximum(0.5, 1.0 / (1.0 + 10.0 ** (-x / 400.0)))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    c = data["correct"] == 1
    return float(np.mean(np.where(c, np.log(p), np.log(1 - p))))


def fit_anchor(data: np.ndarray, scale: float = 400.0, floor2afc=False) -> float:
    """The shift item ratings need, by golden-section over the likelihood."""
    lo, hi = ANCHOR_RANGE
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = (log_likelihood(data, x, scale, floor2afc) for x in (a, b))
    while hi - lo > 0.5:
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = log_likelihood(data, b, scale, floor2afc)
        else:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = log_likelihood(data, a, scale, floor2afc)
    return round((lo + hi) / 2, 1)


def bootstrap(data: np.ndarray, draws: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fits = [fit_anchor(data[rng.integers(0, len(data), len(data))]) for _ in range(draws)]
    return tuple(np.quantile(fits, [0.025, 0.975]))


def report(data: np.ndarray, label: str, draws: int) -> None:
    actual = data["correct"].mean()
    expected = _expected(data["u"], data["i"]).mean()
    print(f"\n{label}: {len(data)} answers Elo scored after calibration")
    print(
        f"  actual {actual:.3f}  model expected {expected:.3f}  shortfall {actual - expected:+.3f}"
    )
    anchor = fit_anchor(data)
    line = f"  anchor: items are {anchor:+.1f} points off the model's belief"
    if draws:
        lo, hi = bootstrap(data, draws)
        line += f"  (95% CI [{lo:+.1f}, {hi:+.1f}], {draws} resamples)"
    print(line)
    print("  by user rating (a non-flat anchor is a shape error no constant fixes):")
    print("    band          n  actual  expect   anchor")
    for lo_r in range(0, 3600, 400):
        band = data[(data["u"] >= lo_r) & (data["u"] < lo_r + 400)]
        if len(band) < MIN_BAND:
            continue
        a, e = band["correct"].mean(), _expected(band["u"], band["i"]).mean()
        print(
            f"    {lo_r:>4}-{lo_r + 400:<4} {len(band):>6}   {a:.3f}   {e:.3f}"
            f"  {fit_anchor(band):+7.1f}"
        )


def link_report(data: np.ndarray) -> None:
    """Accuracy against the served offset, and the two link families' fits.

    The offset spread all comes from the tails — jitter is ±75, so post-
    calibration rows cluster at the target and it is shared answers and freshly
    calibrated users that reach anywhere else. Expect the bins to thin fast;
    what matters is whether the thin ones sit on one curve and off the other.
    """
    print("\naccuracy by (user - item) at scoring, against both links (anchor fitted each):")
    a_elo = fit_anchor(data)
    a_2afc = fit_anchor(data, floor2afc=True)
    print("    offset          n  actual  elo+floor  2afc")
    edges = [-10000, -200, 0, 100, 200, 300, 400, 10000]
    off = data["u"] - data["i"]
    for lo, hi in itertools.pairwise(edges):
        rows = data[(off >= lo) & (off < hi)]
        if not len(rows):
            continue
        x = rows["u"] - (rows["i"] + a_elo)
        p_elo = np.maximum(0.5, 1.0 / (1.0 + 10.0 ** (-x / 400.0))).mean()
        x = rows["u"] - (rows["i"] + a_2afc)
        p_2afc = (0.5 + 0.5 / (1.0 + 10.0 ** (-x / 400.0))).mean()
        print(
            f"    {lo:>6}..{hi:<6} {len(rows):>5}   {rows['correct'].mean():.3f}"
            f"      {p_elo:.3f}  {p_2afc:.3f}"
        )
    print("  mean log-likelihood (higher is better), best (anchor, scale) per family:")
    for name, floor2afc in (("elo+floor", False), ("2afc", True)):
        best = max(
            ((s, fit_anchor(data, s, floor2afc)) for s in range(100, 1300, 100)),
            key=lambda sa: log_likelihood(data, sa[1], sa[0], floor2afc),
        )
        ll400 = log_likelihood(data, fit_anchor(data, 400, floor2afc), 400, floor2afc)
        print(
            f"    {name:<10} at scale 400: {ll400:+.4f};"
            f" free scale {best[0]}: {log_likelihood(data, best[1], best[0], floor2afc):+.4f}"
            f" (anchor {best[1]:+.1f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the user-item scale offset from responses.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="a COPY of the live database")
    ap.add_argument("--link", action="store_true", help="ask the record about the link shape")
    ap.add_argument("--min-ms", type=int, default=0, help="drop answers faster than this")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"{args.db} does not exist")
    conn = open_connection(args.db)

    data = elo_scored(load(conn))
    if not len(data):
        raise SystemExit("no scored responses to fit on")
    fast = data[(data["ms"] >= 0) & (data["ms"] < 3000)]
    if len(fast):
        print(
            f"{len(fast)} answers under 3s ({len(fast) / len(data):.1%}) scoring"
            f" {fast['correct'].mean():.3f} — chance means click-through; --min-ms holds them out"
        )
    if args.min_ms:
        data = data[data["ms"] >= args.min_ms]

    # Snapshots on either side of a scale change are beliefs off different
    # scales, so each era is fitted alone. Splitting on the timestamp is right
    # because the re-derivation and the marker land in one migration.
    boundaries = [
        (key, row["value"])
        for key in ("regraded_at", "anchored_at")
        if (row := conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone())
    ]
    eras = [("all rows", np.ones(len(data), bool))]
    if boundaries:
        eras = []
        starts = [("start of record", ""), *boundaries]
        for (name, since), (until_name, until) in zip(
            starts, [*boundaries, ("now", "9999")], strict=True
        ):
            mask = (data["at"] >= since) & (data["at"] < until)
            if mask.any():
                eras.append((f"{name} → {until_name}", mask))
    for label, mask in eras:
        report(data[mask], label, args.bootstrap)
    print(
        f"\nselection aims {-_TARGET_OFFSET:.0f} below the user for"
        f" {TARGET_ACCURACY:.0%}; a fitted anchor near zero means it gets it"
    )
    if args.link:
        # Only the newest era: the link question is about the deployed model,
        # and offsets snapshotted under an older scale would blur the answer.
        link_report(data[eras[-1][1]])


if __name__ == "__main__":
    main()
