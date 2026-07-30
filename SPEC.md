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
  pre-answer — and that has to hold against a caller who isn't using the UI,
  so the answer key is reachable only by answering a trial the server actually
  offered. Which of the two moves is correct is a coin flip, so it comes from a
  CSPRNG: a predictable one is a leak with extra steps. Enforcement stops at
  making the answer cost an answer, though — someone determined to fool
  themselves through the ordinary API always can, and spending much to prevent
  that would be spending it on the wrong threat.
- **Which arrow belongs to which button is said only in colour**, so that
  pair has to survive red-green colour blindness and the warm filters
  night-mode displays apply, and has to stay free of any hint that one
  move is the safer one. The reveal's best-against-worse pair is held to
  the same standard as far as a light board allows, but there colour
  reinforces a verdict the text already gives.
- **The correct answer is the position's best move** by full-strength
  engine analysis — never a weakened engine, and never "the less bad of two
  bad moves".
- **The distractor is the move actually played in the game** whenever it
  wasn't best (real human errors), with the engine's second choice as
  fallback.
- **Difficulty lives in win-probability space**, never raw centipawns, and
  is a property of the item alone — fixed by the engine's gap when the item
  is labeled, and never revised by anyone's answers. A gap alone can't
  predict how hard a comparison *feels*, so this is knowingly approximate;
  the correction belongs in offline analysis, which can regularise it and
  isn't the thing choosing which items get served. Online it would make
  every user's difficulty a function of every other user's answers, which
  costs more — in coupling, and in a feedback loop that biases the very
  estimate it produces — than the targeting accuracy is worth.
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
- **Nothing gates the first trial.** Arriving is a read: no name to type, and
  nothing written, so there is nothing about arriving that could need
  rationing. Identity is issued anonymously by the first *answer* — which is
  also the first thing worth keeping — and an account is optional and, when
  created, claims the history already earned rather than starting a fresh one.
  A user's row must never be reachable by guessing a name.
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
  would read it as. Nothing derived from them survives either — difficulty
  comes from the engine, not from answers — so "erased" needs no asterisk
  beyond the backup window.
- **Explanations are grounded in engine output.** The engine lines are the
  authority; any prose (user's pasted questions today, generated
  narration later) sits next to them, never replaces them.
- **The record outlives the deployment.** Responses are the experiment, so
  they are replicated off the machine that serves them, and no routine
  operation — refreshing the item bank especially — may replace the live
  database wholesale.

## Not yet built

- Color/tone overlay during the reveal (the synesthesia hypothesis this
  project grew out of) and Stroop-interference measurement of automaticity.
- Automatic LLM narration of the engine lines.
- Password reset (the optional email is stored for it but nothing sends
  mail yet).
- Transfer measurement: in-app accuracy shares the item generator's
  biases; the real test is external (rated games, or items from a
  deliberately different distribution).
- A learning-rate measure that survives adaptive difficulty. Selection
  holds accuracy near 80% by design, so raw accuracy is flat whatever the
  user is doing; the rate has to come from the difficulty being sustained,
  and needs item difficulty estimated offline rather than assumed from the
  gap. Issue #27 has the model and the probe trials it wants.
