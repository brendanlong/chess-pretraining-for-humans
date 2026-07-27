# chess-pretraining-for-humans

A pairwise move-discrimination trainer: you're shown a real position from a
real game and two candidate moves, and you pick the better one. Stockfish is
the ground truth; feedback is immediate; difficulty adapts to hold you near
80% accuracy.

The framing is *supervised pretraining for humans*: instead of learning chess
from sparse end-of-game reward, train the underlying perceptual judgment —
"this move is better than that move" — with thousands of dense, labeled,
fast trials, the way perceptual category learning works in chicken sexing or
radiology. Forced choice plus immediate feedback is the load-bearing
mechanism.

## Design

- **Correct answer = the position's best move** (full-strength Stockfish,
  multipv). Discriminating between two bad moves isn't a useful skill, and
  being wrong against real truth beats learning to prefer weaker moves.
- **Distractor = the move actually played in the game** when it wasn't best
  (those are the errors humans actually make); the engine's second choice as
  fallback when the game move was best.
- **Difficulty lives in win-probability space**, not centipawns (a 1.0-pawn
  gap at 0.00 is enormous; at +6.00 it's noise). Evals are converted with
  Lichess's logistic model, and the wp-gap seeds a per-item Elo rating that
  real responses then correct — gap alone can't hold a target accuracy.
- **Learnability filter**: both moves are re-evaluated at shallow depth. If
  shallow and deep search disagree about which move is better, the answer
  hinges on deep calculation rather than anything perceivable, so the item is
  labeled correct-but-unlearnable and never served.
- **Probe trials**: every 8th trial gives no feedback. Those are the real
  measure — the failure mode to guard against is learning to read the
  feedback loop rather than the board.

## Running it

Requires [uv](https://docs.astral.sh/uv/), a `stockfish` binary on PATH, and
`zstd`.

```bash
# 1. Mine candidate positions from a Lichess monthly dump (streams the head,
#    no need to download 30GB; only analyzed games are parsed)
curl -s -r 0-150000000 https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst \
  | zstdcat 2>/dev/null \
  | uv run python -m trainer.mine --max-candidates 2500 > data/candidates.jsonl

# 2. Label with Stockfish (deep multipv for ground truth, shallow pass for
#    the learnability filter) — builds data/items.db
uv run python -m trainer.label data/candidates.jsonl

# 3. Serve
uv run uvicorn trainer.server:app --host 0.0.0.0 --port 8000
```

Open the page, press <kbd>1</kbd>/<kbd>2</kbd> to answer,
<kbd>space</kbd> for the next trial. `?user=name` keeps separate profiles.

## Layout

| path | what |
|---|---|
| `trainer/mine.py` | Lichess dump stream → candidate decision points |
| `trainer/label.py` | Stockfish labeling: best move, evals, learnability, seed rating |
| `trainer/winprob.py` | centipawns → win probability |
| `trainer/rating.py` | Elo machinery: item selection targets ~80% expected score |
| `trainer/server.py` | FastAPI: `/api/next`, `/api/answer`, `/api/stats` |
| `web/` | vanilla-JS frontend on [chessground](https://github.com/lichess-org/chessground) (vendored) |

## Not built yet

- The color/sound overlay during reveal (the synesthesia hypothesis — v1.1).
- Stroop-interference measurement (deliberately mismatched cue) and transfer
  measurement; probe-trial accuracy is the v1 stand-in.
- Glicko-2 (plain Elo for now).
