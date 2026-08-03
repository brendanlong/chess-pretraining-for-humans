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
  ray, where the arrows coincide and so do the numbers. The discs on the
  arrows can be turned off, which is the other reason the colour pair has to
  stand on its own rather than leaning on them. Neither channel may hint that one move is
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
  as the deep gap does. It does not quite subsume it — the deep gap still
  carries a little signal of its own — but what it carries has no consistent
  direction, flipping sign from band to band, so there is nothing to add it
  as. It is a property of the item alone — fixed when
  the item is labeled, and never revised by anyone's answers. Still knowingly
  approximate; the correction belongs in offline analysis, which can
  regularise it and isn't the thing choosing which items get served. Where
  the scale *sits* is held to the same standard as its slope: measured, not
  chosen — the slope from the strength of the players whose errors the items
  are, the location from the accuracy real users produce against it, refit
  offline as a constant and never nudged by any single answer. Online
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
  not. The fit must also be taken on the half of the bank nobody aimed at a
  band: including the half that was, which was mined at deliberately chosen
  gaps, moves the same measurement by a factor of three. That half is still
  mined through a gap window — nothing here is a random sample of errors —
  but a window is not a target.
- **An item whose answer the engine won't hold still is never served.** Not a
  difficulty judgement — the scale reaches as far as the engine can see, so
  nothing is dropped for being hard. It is that the search which picked the
  best move and a search restricted to the two candidates can disagree about
  which is better, and an item on both sides of that disagreement has nothing
  to teach. The reading has to be taken on an engine that has not just been
  told the answer — a shallow search reading a deep search's leftovers is not
  a shallow search — and it has to be reproducible, or an item's difficulty
  is a coin flip made once.
- **Selection never repeats an item while fresh ones remain**, so every
  answer it chooses is a first exposure recorded before feedback —
  simultaneously a clean measurement and a training trial. The two things
  that do serve a repeat are the bank running out and a URL naming a
  position this user has answered; both are flagged, and a flagged answer
  is excluded from ratings and accuracy, which is what makes asking for
  one harmless. The remedy for exhaustion is mining more games.
- **Adapting difficulty is a promise about the bank, not just about
  selection.** Selection cannot fail — it serves the nearest items it
  holds, however far off they are — so a difficulty the bank is thin at
  produces no error and no signal, only users quietly held at the wrong
  accuracy. Every rating a user can occupy has to have enough items near
  it to spend a session inside, which is a thing to measure rather than
  assume. Where it can't be met it is named, and it is the *easy* end
  that can't be: a position a shallow look settles at a glance is one a
  human rarely got wrong, so few of them were ever mined, and the bands a
  beginner is aimed at are the thin ones. The remedy has to stay a real one
  — mining, not a constant — which means the thin bands must be reachable
  from what the pipeline can filter on. They are: the deep gap is what
  mining steers, it correlates with difficulty at 0.79, and a wide window
  at the blundering end lands about half of what it labels in the region
  that is short. That is a shotgun and not a rifle, so the order is priced
  per region rather than per band, and the supply report has to say where a
  window actually landed rather than pretend the aim was true.
- **New users are assumed to be beginners** — no strength question; the
  rating system must instead climb fast for experienced players.
- **Nothing gates the first trial.** Arriving is a read: no name to type, and
  nothing written, so there is nothing about arriving that could need
  rationing. Identity is issued anonymously by the first *answer* — which is
  also the first thing worth keeping — and an account is optional and, when
  created, claims the history already earned rather than starting a fresh one.
  A user's row must never be reachable by guessing a name.
- **An answer somebody committed to is not thrown away for a reason the
  server can fix.** The trial's token is short-lived, so a tab that sat
  through lunch comes back holding one the clock has run out on — which says
  nothing about who is holding it or what they were offered. That one is
  re-signed, and the pick the user actually made is what gets recorded,
  timed by the decision rather than by the round trip that rescued it. The
  refusals that stand are the ones re-signing would have to guess at: a
  trial issued to a session that has since changed, where replaying would
  file one person's answer under another, and a token this process can no
  longer verify, which has nothing left to carry over.
- **Responses are research data, and the notice says so on the page that
  records them** — nothing gates the first trial, so consent can't be
  collected before it; what's owed is that a guest never has to go
  looking to find out. The published record is per-user random ids,
  answers and timing, and (once Lichess linking exists) a rating band
  rather than a number — never usernames or emails. What the privacy
  policy promises is a constraint on what the analysis may export, not
  just prose.
- **The page counter learns that a page was opened, and nothing else.**
  The address bar names the position on screen, so nothing read off the URL
  may reach it: what it is sent is a path chosen from a closed list of pages,
  never the URL, the query string, the title, or a referrer from our own
  origin. A page missing from that list counts as nothing, which is the
  failure a sanitizer can't have — that one is always an unanticipated URL
  away from the other. Building the request ourselves is also what keeps a
  third party's script off an origin that holds a session cookie: what the
  privacy policy says about the counter is then enforced by code that ships
  with the app rather than trusted to code that doesn't.
- **A shared position is an ordinary trial that says it was shared.** Once you
  have answered one, the address bar names it, so sending it to a friend is
  copying a URL, and what arrives is the same symmetric trial everyone else
  gets — carrying an item id and never an answer. It names it only *after* the
  answer, because a URL that named the trial in progress would be asking the
  server for it on every reload, and marking a position the app itself chose
  as one nobody aimed. A URL is a durable thing that gets reopened — the tab
  reloads, the link comes back around, a friend sends it on — so one naming an
  item this user has already answered opens *that* position again rather than
  a stranger's, as a rerun: answerable and explained, and worth nothing to the
  rating. That is what makes serving a position whose answer the user may
  remember safe, and it is the same bargain the exhausted bank already
  strikes. On a first exposure the
  answer counts: it rates and it is counted in accuracy, because it is a real
  first exposure against an item of measured difficulty, and Elo is a
  statement about the item and not about who chose it. What it carries is a
  mark, so the analysis can hold out the trials nobody aimed. The one thing
  that can't take an unaimed item is the calibration staircase, which steps
  by a fixed amount *because* selection guarantees the item was aimed: on
  somebody else's position it would pay a quarter of the scale for what, in a
  two-alternative task, a beginner wins half the time by guessing. So during
  calibration a shared answer is scored by Elo, which reads how hard the item
  actually was, instead.
- **What is held is downloadable, and it is the same set deletion erases.**
  The privacy policy describes the record in prose, and prose drifts; a file
  that *is* the record cannot, so the honest answer to "what do you have on
  me" is a copy rather than a paragraph about one. Tying it to the deletion
  promise is the point — if the two ever name different sets, one of them is
  lying. The exceptions are the password hash — which exists so that nobody
  holds the password, and so is not the user's to have back — and internal
  bookkeeping, like row ids, that says nothing about the user. Being signed in
  is the whole authorization, because it is already the whole authorization
  for reading the same record inside the app: asking for a password would put
  a wall in front of a guest, who has none, and whose copy is the only thing
  that survives clearing the cookie.
- **A deletion request erases the responses too**, and is reachable from
  inside the app by whoever the record belongs to — which includes a guest,
  because a guest's answers are the same research data an account's are.
  Holding the session is the proof of ownership the optional, unverified
  email can't supply, so deletion can't depend on an email thread. An
  account is asked for its password on top, since a shared browser holds the
  session; a guest has none, and no second factor exists to invent for one,
  so the cookie has to be enough. The alternative is that the guest half of
  what is held has no erase button at all and "clear the cookie" stands in
  for one — and that makes the record unreachable, which is not the same as
  gone. Keeping the answers while dropping the name would leave us holding
  data the user believes is gone; losing some responses is cheaper than
  making "we deleted it" mean something narrower than a user would read it
  as. Nothing derived from them survives either — difficulty comes from the
  engine, not from answers — so the only asterisks left are the ones no
  design can remove: the backup window, and analysis already published.
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
