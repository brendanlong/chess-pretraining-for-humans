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

Each is stated as a rule with the short reason it holds. The longer
reasoning lives where DESIGN.md and CALIBRATION.md point.

### The trial

- **Nothing may leak the answer before the user commits.** No UI element,
  API payload, timing, or (future) overlay cue may distinguish the two moves
  pre-answer, and that holds against callers who aren't using the UI: the
  answer key is reachable only by answering a trial the server offered.
  Which move sits in which slot comes from a CSPRNG, since a predictable one
  is a leak. Enforcement stops at making the answer cost an answer; someone
  set on fooling themselves through the API always can.
- **Which arrow belongs to which button is said twice**, in colour and in a
  number both carry, and neither may hint that one move is the safer one.
  The colour pair must stand on its own — the numbered discs can be turned
  off — so it has to survive red-green colour blindness and warm night-mode
  filters. The reveal's best-against-worse pair meets the same standard as
  far as a light board allows.
- **The correct answer is the position's best move** by full-strength
  engine analysis — never a weakened engine, and never "the less bad of two
  bad moves".
- **The distractor is the move actually played in the game** whenever it
  wasn't best (real human errors), with the engine's second choice as
  fallback.

### Difficulty

- **Difficulty is what a shallow search could see, not what the answer is
  worth.** It is read in win-probability space from the gap the engine's
  first few plies found, not the gap at full depth, because the question put
  to a human is how much there was to notice. That is also why a position
  whose surface recommends the losing move — a negative shallow gap — must
  rate as the hardest kind there is.
- **An item's difficulty is fixed when it is labeled** and never revised by
  anyone's answers. The curve's slope and location are measured, not
  chosen — from the strength of the players who made these errors, on the
  half of the bank that wasn't mined at chosen gaps, and from the accuracy
  users produce — and refit offline as constants; online correction would
  couple every user's difficulty to every other user's answers.
- **Lookahead is part of what adapts, not a cutoff.** Measurement runs as
  deep as the ground-truth search, and it is the rating, not a filter, that
  keeps a hard item from someone who couldn't have seen it.
- **The whole per-depth measurement is kept**, not just its summary, so a
  better difficulty model can be fitted without re-labeling the bank.
- **Any two items must be orderable by difficulty.** The curve may be
  approximate but never flat, and saturates rather than stops outside the
  range the evidence covers: a wrong ordering is recoverable, no ordering is
  not.
- **An item whose full-depth verdict the engine won't hold is never
  served.** If the search that picked the best move and a search restricted
  to the two candidates disagree at full depth, the item has nothing to
  teach. That reading is taken from a cleared hash and must be
  reproducible.

### Selection

- **Selection never repeats an item while fresh ones remain**, so every
  chosen answer is a first exposure, both measurement and training. The only
  repeats are an exhausted bank and a URL naming a position this user has
  answered; both are flagged and excluded from ratings and accuracy.
- **Every rating a user can occupy must have enough items near it** to spend
  a session inside. Selection can't fail — it serves the nearest items
  however far off — so a thin band shows up only as users held at the wrong
  accuracy, and has to be measured rather than assumed. The remedy is
  mining, which means thin regions must stay reachable from what mining can
  filter on.
- **New users are assumed to be beginners** — no strength question; the
  rating system must instead climb fast for experienced players.
- **A shared position is an ordinary trial that says it was shared.** The
  URL names a position only after it is answered, and carries an item id,
  never an answer. A first exposure to a shared item rates and counts
  normally but is marked, so analysis can hold out trials nobody aimed;
  during calibration it is scored by Elo rather than the staircase, which
  assumes an aimed item. Reopening one already answered serves it as a
  repeat.

### Identity and data

- **Nothing gates the first trial.** Arriving writes nothing; identity is
  issued anonymously by the first answer. An account is optional and claims
  the history already earned. A user's row must never be reachable by
  guessing a name.
- **An answer the user committed to is not discarded for a reason the
  server can fix.** An expired trial token is re-signed and the original
  pick and timing recorded. The refusals that stand are a token issued to a
  session that has since changed, and one this process can no longer verify.
- **Responses are research data, and the page that records them says so.**
  Consent can't precede the first trial, so a guest must never have to go
  looking. The published record is per-user random ids, answers, timing and
  rating snapshots, plus (with Lichess linking) a Lichess rating band, never
  the number — and never usernames or emails. The privacy policy constrains
  what analysis may export.
- **The page counter learns only that a page was opened.** It is sent a
  path from a closed list of pages — never the URL, query, title, or
  same-origin referrer — and a page missing from the list counts as
  nothing. No externally hosted script runs on the origin.
- **What is held is downloadable, and it is the same set deletion erases**,
  so the two can't drift apart. The exceptions are the password hash and
  internal bookkeeping like row ids. Holding the session is the whole
  authorization, guests included; no password is asked.
- **Deletion erases the responses too**, and is reachable from inside the
  app by whoever the record belongs to, guests included — holding the
  session is the proof of ownership. Accounts also confirm with their
  password. Nothing derived from responses survives; the remaining caveats
  are the backup window and analysis already published.
- **Explanations are grounded in engine output.** The engine lines are the
  authority; any prose sits next to them, never replaces them.
- **The record outlives the deployment.** Responses are replicated off the
  serving machine, and no routine operation — refreshing the item bank
  especially — may replace the live database wholesale.

## Not yet built

- Color/tone overlay during the reveal (the synesthesia hypothesis this
  project grew out of) and Stroop-interference measurement of automaticity.
- Automatic LLM narration of the engine lines.
- Password reset (the optional email is stored for it but nothing sends
  mail yet).
- Lichess account linking; what it would add to the published record — a
  rating band, never a number — is already committed to above and in the
  privacy policy.
- Transfer measurement: in-app accuracy shares the item generator's
  biases; the real test is external (rated games, or items from a
  deliberately different distribution).
- A learning-rate measure that survives adaptive difficulty. Selection
  holds accuracy near 80% by design, so raw accuracy is flat whatever the
  user is doing; the rate has to come from the difficulty being sustained,
  and needs item difficulty estimated offline rather than assumed from the
  gap. Issue #27 has the model and the probe trials it wants.
