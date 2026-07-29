# SPEC — what this is trying to do

## Goal

Train the perception of chess move quality directly, using dense supervised
feedback: thousands of fast, labeled, forced-choice trials instead of the
sparse end-of-game reward chess normally provides. This is a proof of
concept for human uplift via supervised pretraining — the model is
perceptual category learning (chicken sexing, radiology), where volume of
labeled trials with immediate feedback produces fast, automatic,
hard-to-articulate judgment. Expected to help beginners most; at low skill
the task reduces to "spot the blunder", and the target is noticing blunders
automatically.

## Core loop

- Show a real position from a real game and two candidate moves.
- The user picks the better move; the reveal shows which the engine
  prefers, both evaluations, and both engine lines, replayable on the board.
- Difficulty adapts to hold the user near 80% accuracy.
- Pace is meant to be fast — first-instinct answers, with attention spent
  on the reveals that surprise you, not on pre-answer calculation.

## Invariants

- **Nothing may leak the answer before the user commits.** No UI element,
  API payload, or (future) overlay cue may distinguish the two moves
  pre-answer.
- **The correct answer is the position's best move** by full-strength
  engine analysis — never a weakened engine, and never "the less bad of two
  bad moves".
- **The distractor is the move actually played in the game** whenever it
  wasn't best (real human errors), with the engine's second choice as
  fallback.
- **Difficulty lives in win-probability space**, never raw centipawns, and
  is corrected per-item from real responses (a gap alone can't predict how
  hard a comparison feels).
- **Items must be learnable**: if shallow and deep search disagree about
  which move is better, the answer isn't reachable from the surface and
  the item is never served.
- **Items never repeat while fresh ones remain**, so every answer is a
  first exposure recorded before feedback — simultaneously a clean
  measurement and a training trial. Repeats (bank exhausted) are flagged
  and excluded from ratings and accuracy. The remedy for exhaustion is
  mining more games.
- **New users are assumed to be beginners** — no strength question; the
  rating system must instead climb fast for experienced players.
- **Nothing gates the first trial.** Identity is issued automatically and
  anonymously on arrival; an account is optional and, when created, claims
  the history already earned rather than starting a fresh one. A user's row
  must never be reachable by guessing a name.
- **Responses are research data, and the notice says so on the page that
  records them** — nothing gates the first trial, so consent can't be
  collected before it; what's owed is that a guest never has to go
  looking to find out. The published record is per-user random ids,
  answers and timing, and (once Lichess linking exists) a rating band
  rather than a number — never usernames or emails. What the privacy
  policy promises is a constraint on what the analysis may export, not
  just prose.
- **A deletion request erases the responses too**, and is reachable from
  inside the app. Being signed in is the proof of ownership the optional,
  unverified email can't supply, so deletion can't depend on an email
  thread; and keeping the answers while dropping the name would leave us
  holding data the user believes is gone. Losing some responses is
  cheaper than making "we deleted it" mean something narrower than a user
  would read it as. What survives is each item's answered/correct counters
  and the difficulty they feed: not reversible, not attributable to
  anyone, and the policy says so rather than rounding it up to "erased".
- **Explanations are grounded in engine output.** The engine lines are the
  authority; any prose (user's pasted questions today, generated
  narration later) sits next to them, never replaces them.

## Not yet built

- Color/tone overlay during the reveal (the synesthesia hypothesis this
  project grew out of) and Stroop-interference measurement of automaticity.
- Automatic LLM narration of the engine lines.
- Password reset (the optional email is stored for it but nothing sends
  mail yet).
- Transfer measurement: in-app accuracy shares the item generator's
  biases; the real test is external (rated games, or items from a
  deliberately different distribution).
