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
- **Which arrow belongs to which button is said twice**: in colour, and in
  a number carried by both. The colour pair still has to survive red-green
  colour blindness and the warm filters night-mode displays apply, because
  it is what the eye reads at a glance; the number is what remains when it
  doesn't, and when the two arrows cross or point at the same square and
  no colour would have separated them — short of the two arriving along one
  ray, where the arrows coincide and so do the numbers. The numbers can be
  turned off, which is the other reason the colour pair has to stand on its
  own rather than leaning on them. Neither channel may hint that one move is
  the safer one. The reveal's best-against-worse pair is held to the same
  standard as far as a light board allows, but there both reinforce a verdict
  the text already gives.
- **The correct answer is the position's best move** by full-strength
  engine analysis — never a weakened engine, and never "the less bad of two
  bad moves".
- **The distractor is the move actually played in the game** whenever it
  wasn't best (real human errors), with the engine's second choice as
  fallback.
- **Difficulty is what a shallow search could see, not what the answer is
  worth.** It lives in win-probability space, never raw centipawns — but the
  gap it reads is the one the first few plies of the engine's own search
  found, averaged, and not the gap at full depth. The deep gap says how much
  the mistake costs; this says how much there was to notice, which is the
  thing a human is being asked to do. Measured against the strength of the
  players who really made these errors it predicts about half again as much
  as the deep gap does, and it subsumes it: once the shallow gap is known,
  the deep one adds nothing. It is a property of the item alone — fixed when
  the item is labeled, and never revised by anyone's answers. Still knowingly
  approximate; the correction belongs in offline analysis, which can
  regularise it and isn't the thing choosing which items get served. Online
  it would make every user's difficulty a function of every other user's
  answers, which costs more — in coupling, and in a feedback loop that biases
  the very estimate it produces — than the targeting accuracy is worth.
- **A position whose surface recommends the losing move is the hardest kind
  there is, and the scale has to say so.** That is what a *negative* shallow
  gap is, and about one item in twenty is one. It is also why the axis can't
  be the deep gap: some of the most lopsided positions in the bank, worth
  half a win to get right, are ones a shallow look gets backwards — and on
  the deep gap those rate as the easiest items there are, which is exactly
  who they were being served to.
- **How far ahead a user has to look is part of what adapts, and nothing is
  thrown away for needing too much of it.** Lookahead is a difficulty, so it
  belongs on the rating rather than in a cutoff that is the same for
  everybody: a beginner should be spending their attention on what a position
  *looks* like before spending it on reading four moves into one. The
  measurement therefore runs as far as the engine's own ground-truth search,
  and it is the rating, not a filter, that keeps a hard item away from
  someone who couldn't have seen it. A cutoff would be the thing this exists
  to replace.
- **The measurement is kept, not just its summary.** The search that produces
  it is the expensive half of labeling and every way of reading it is cheap,
  so the whole per-depth curve is stored. A better difficulty model can then
  be fitted without going near an engine again — which is the difference
  between an afternoon and re-labeling the bank.
- **Any two items a bank can hold must be orderable by difficulty.** The
  curve may be approximate, but it may not be flat: a range of items that all
  map to one rating is a range selection cannot aim inside, and it lands on
  whichever users sit there. What the curve is worth is measured, from the
  strength of the humans whose real errors the items record — and only over
  the range where that measurement has any power. Outside it, at both ends,
  the curve saturates rather than stopping, so gaps nothing speaks to stay
  ordered and distinct. A wrong ordering is recoverable and no ordering is
  not. The fit must also be taken on errors nobody selected for: half this
  bank was mined at deliberately chosen gaps, and including it moves the same
  measurement by a factor of three.
- **An item whose answer the engine won't hold still is never served.** Not a
  difficulty judgement — the scale reaches as far as the engine can see, so
  nothing is dropped for being hard. It is that the search which picked the
  best move and a search restricted to the two candidates can disagree about
  which is better, and an item on both sides of that disagreement has nothing
  to teach. The reading has to be taken on an engine that has not just been
  told the answer — a shallow search reading a deep search's leftovers is not
  a shallow search — and it has to be reproducible, or an item's difficulty
  is a coin flip made once.
- **Items never repeat while fresh ones remain**, so every answer is a
  first exposure recorded before feedback — simultaneously a clean
  measurement and a training trial. Repeats (bank exhausted) are flagged
  and excluded from ratings and accuracy. The remedy for exhaustion is
  mining more games.
- **Adapting difficulty is a promise about the bank, not just about
  selection.** Selection cannot fail — it serves the nearest items it
  holds, however far off they are — so a difficulty the bank is thin at
  produces no error and no signal, only users quietly held at the wrong
  accuracy. Every rating a user can occupy has to have enough items near
  it to spend a session inside, which is a thing to measure rather than
  assume. Where it can't be met it is named, and it is now the *easy* end
  that can't be: a position a shallow look settles at a glance is one a
  human rarely got wrong, so few of them were ever mined, and the bands a
  beginner is aimed at are the thin ones. The remedy is mining — but only
  indirectly, because what mining can filter on is the deep gap and what
  difficulty is made of is the shallow one. The two correlate and no more
  than that, so an order placed in deep gaps scatters across bands, and the
  supply report has to say where a bin actually landed rather than pretend
  the aim was true.
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
- **The page counter learns that a page was opened, and nothing else.**
  It is sent the URL, the query string and the title, so what keeps the
  research record out of it is that the app writes none of the three — an
  item id in the URL would ship answers off-site without touching a line
  that looks like it is about privacy. Being a third party's script on an
  origin that holds a session cookie, it is pinned to a hash: what the
  policy says about it is enforced, not trusted.
- **A deletion request erases the responses too**, and is reachable from
  inside the app. Being signed in is the proof of ownership the optional,
  unverified email can't supply, so deletion can't depend on an email
  thread; and keeping the answers while dropping the name would leave us
  holding data the user believes is gone. Losing some responses is
  cheaper than making "we deleted it" mean something narrower than a user
  would read it as. Nothing derived from them survives either — difficulty
  comes from the engine, not from answers — so the only asterisks left are
  the ones no design can remove: the backup window, and analysis already
  published.
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
