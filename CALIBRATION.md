# CALIBRATION — what difficulty is measured from, and what is still open

SPEC says difficulty must be measured rather than chosen. This is the record
of the measuring: what evidence exists, what it can and can't answer, what was
tried and rejected, and what to re-run before changing a constant. The
constants themselves justify their own values where they live, in
`trainer/rating.py` and `trainer/label.py`; nothing here repeats a number that
a comment beside the code already owns.

## The evidence

One thing, and it is thinner than it looks: **every `game`-source item is a
mistake a real human made, and `mover_elo` says how strong they were.** That
pair — an item's measurements against the strength of the player who got it
wrong — is the only external signal in the project. Everything else in the
bank is engine output, which can say what is true but not what is hard.

Three measurements per item feed it, all from `trainer/label.py`:

- `gap_wp` — the win-probability gap at full depth. What the answer is worth.
- `gap_ladder` — that gap at every depth from 1 up, from one search restricted
  to the two candidate moves. What there was to *see*, at each amount of
  looking. This is the expensive column and the reason the others can be
  recomputed cheaply.
- `shallow_gap` — the ladder's shallow end, averaged. Difficulty is a function
  of this and nothing else.

Two properties of the sample decide what may be done with it.

**Only errors are ever recorded.** Mining sees a position because somebody
blundered there; it never sees the move that wasn't a blunder. So there is no
denominator, and therefore no way to ask "how often does a 1500 miss this" —
only "how big is the mistake a 1500 still makes". Every fit here is a boundary
read off a quantile, never a rate.

**Half the bank was mined at chosen gap bands and half was not.** The
untargeted half is the 12,660 positions in the original seed bank; the rest
were mined with `--min-gap-wp`/`--max-gap-wp` aimed at filling holes, which is
selection on the very quantity being regressed. Fitting on everything moves the
same measurement by a factor of three. Which window a position came through is
recorded on it (`items.mined_untargeted`), so the fit restricts itself and there
is no reference bank to keep. `--everything` lifts the restriction, which is how
to see what the selection is worth rather than taking the factor on faith.

## The estimator, and why it's this one

Bin errors by player strength, take the 75th percentile of the gap in each
band, fit strength back against that. `trainer/fit_difficulty.py` is the
implementation and its docstrings carry the reasoning for each choice. It uses
numpy, from the dev group and not the deployment, because the published
constants were fitted with it: a quantile that interpolates differently moves
the slope by percent, which reads as a disagreement with a comment when it is
only a convention.

The direction is the part that is easy to get wrong and worth stating twice:
this asks *how big an error a player of a given strength still makes*, not *how
strong the player who made an error of a given size was*. Those are the two
regressions of one weak relationship and **they differ by a factor of about
twenty** — 6096 against 297, measured on the deep gap. There
is no fact of the matter between them without a model of erring that nobody
has. The project picked one and has to keep picking it, because a constant
fitted one way is not comparable with one fitted the other. That ambiguity is
the reason a second axis's *magnitude* can never be measured off this data,
only its sign and shape — see the rejected designs below.

To compare candidate axes on equal terms, `--axes` scores each by how far the
tail of erring strength moves from its easiest to its hardest bin. That is
scale-free, so it can rank measures that have no curve yet.

## Rejected, and why

The order matters: each was tried against the one before it.

**The deep gap alone.** The original axis. It works, and it is what `GAP_SLOPE`
was first fitted on — but it answers "what does this mistake cost", and the
question a trainer needs is "what was there to notice". Its worst failure is
concrete: the most lopsided positions in the bank, worth half a game, include
ones a shallow search gets *backwards*, and it rated those among the easiest
items there are.

**Required lookahead, as a second axis beside the gap.** The shallowest depth
from which the pair stays correctly ordered — still computed, because it is what
decides whether an item is learnable, but no longer a difficulty. Rejected on
two counts. Its magnitude is unidentifiable:
read against `GAP_SLOPE` it lands anywhere from 700 to 2400 points a doubling
depending on which of the two regressions above you believe. And it is worth
less than it looks — scored on the same footing as the alternatives it
separates strength about a third as well as the shallow gap does. It survives
as the learnability verdict, which is all it was ever reliably saying.

**A hard cap on lookahead.** Before any of this, items needing more than eight
plies were discarded. That is a difficulty judgement made once for everybody,
which is what issue #48 objected to; the ladder now runs to full depth and
grades what it finds. Almost nothing is dropped as a result — the ground truth
*is* the deepest verdict, so it nearly always settles.

**One search per move, compared.** The first ladder ran a separate search for
each candidate and compared the two numbers. Those come from alpha-beta windows
that never saw each other and disagree with a single ranking search on about a
fifth of positions. A multipv search over just the two moves asks at every
depth the same question the deep pass answers once, and costs less.

**Rescaling the new curve to sit where the old one did**, to avoid regrading
users. Impossible: the shallow-gap axis is intrinsically taller — its
calibrated band alone spans more than the whole of the old one — so any
constant that lowered it far enough pushes the easy tail below zero.

## A second opinion that isn't an engine, and why it can't be scored here

Everything above is read off a search engine, which is why the whole axis
explains so little: an engine can say what is true and not what is hard. A
human-imitation policy conditioned on rating — Maia — can be asked the other
question, and `analysis/maia/` is what asking cost. Nothing is adopted. The
main thing it returned is a warning about the estimator every other line in
this file rests on.

**The estimator cannot evaluate a measure that has seen the played move.**
`spread` scores a candidate axis by how far erring strength moves across it,
so its target is `mover_elo`. The distractor *is* that player's move. So any
statistic that inverts a rating-conditioned model over the distractor is a
rating classifier being scored against rating, and it wins enormously without
saying anything about difficulty.

That is not hypothetical; it is what the first pass here found and reported.
The measure was the *elo-gradient*, Maia's log-odds between the two moves at
1900 minus the same at 1100, and combined with the shallow gap it scored +441
against +326 with a paired interval clear of zero, in both model families and
under every robustness check tried. It decomposes exactly into a term over
each move, and `axes.py` prints both:

- the half over the **best** move — how much more visible the right answer
  gets as skill rises, which is the whole stated mechanism and the only half
  that describes the discrimination the app trains — scores **+59**, and
  *lowers* the combined axis below the shallow gap alone;
- the half over the **played** move, which never looks at the best move at
  all, scores **+447**.

A difficulty measure for a two-alternative choice that ignores one of the two
alternatives is not measuring the choice. `control.py` prices it: on control
positions where the human played the engine's best move — no error, no
distractor, nothing to discriminate — the same statistic still spreads erring
strength by **+196**, three fifths of what the shallow gap achieves on the
whole bank. That is the size of the tautology, and it has to come off any
score before the remainder is difficulty.

The shallow gap is not exposed to this. It is an engine number over a position
and two moves, and the engine has no notion of who was playing, so its
correlation with `mover_elo` has to run through weaker players making more
visible errors. Maia's does not have to.

What survives, none of it a difficulty axis:

- **Maia disagrees with Stockfish constantly, and mostly not about us.** Its
  top move is Stockfish's on 36% of the bank; on a control mined with the gap
  window opened all the way it is 43%, against 37% for the humans who were
  actually there. The disagreement is Maia being a club player, and selecting
  for blunders costs only the seven points between those.
- **It corroborates the negative-shallow-gap items** from outside the ladder.
  Where the surface recommends the losing move, Maia at 1500 prefers the right
  one 24% of the time against 64% over the rest of the bank.
- **The play-versus-recognise gap is still unpriced.** Maia predicts what a
  player would *play*; the app asks what they can *recognise* between two named
  moves, which is easier. This was filed as the caveat that mattered and it is
  not — the circularity above is — but it stands, and the same answer disposes
  of both: `responses` holds human accuracy per item, which is a target no move
  of the player's is an input to. Until this is run against that, the probe has
  measured what Maia knows about ratings and not what makes an item hard.

Reproducibility, if any of it is ever revisited: the two families agree on the
ordering and not the number — 0.86 on the log-odds over the whole bank, 0.69 on
the gradient over the subset `axes.py` fits. Both figures need their subset
named, because the difference between them is mostly castling (see the probe's
README).

## Open, and what to check when changing it

- **Which window.** `SHALLOW_PLIES` is a choice inside the noise, not an
  argmax, and its comment says so. `--windows` prints every window against two
  metrics that disagree about the answer; expect the metric you pick to matter
  more than the value you land on.
- **The easy end is thin and the top is thinner.** `trainer.supply` names the
  bands. The remedy is mining, and README's "keeping the bank full" carries the
  measured yields — the thing to know is that mining steers the deep gap while
  difficulty is the shallow one, so an order is priced per region, not per band.
- **Nothing here uses a single response.** Difficulty is fixed at labeling
  time on purpose (SPEC says why), so the app's own data has never been fed
  back. Issue #27 is the model that would, offline, and `gap_ladder` exists
  partly so that it can be refitted without re-labeling anything.
- **The whole axis explains a few percent of the variance in erring strength.**
  That is not a defect to be fixed by tuning; it is what one engine number can
  say about one human's attention. Treat a change that improves it a lot with
  suspicion, and compare it against `--everything` before believing it.

**Changing a constant moves every item and no user.** `db.connect` re-derives
`items.rating` on the next open, so the bank lands on the new curve by itself —
but nothing moves `users.rating`, and a rating means nothing except against the
difficulties it selects. So a retune has to carry its own regrade: work out what
the new scale does to the item distribution, map each user onto the position
that serves them comparable items, and gate it on a key in `meta` so it cannot
run twice. There is no standing code for this because the mapping
is a property of the change, not of the app.

Re-running the fit is `uv run python -m trainer.fit_difficulty`. If it stops
returning what `rating.GAP_SLOPE`'s comment claims, the estimator has drifted
and every comparison in this file is void — which is why it also prints the
same method applied to the deep gap, which is a fixed point: it returns 6096,
and if that drifts the estimator has changed rather than the data.
