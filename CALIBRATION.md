# CALIBRATION — what difficulty is measured from, and what is still open

SPEC says difficulty must be measured rather than chosen. This is the record
of the measuring: what evidence exists, what it can and can't answer, what was
tried and rejected, and what to re-run before changing a constant. The
constants themselves justify their own values where they live, in
`trainer/rating.py` and `trainer/label.py`; nothing here repeats a number that
a comment beside the code already owns.

## The evidence

Two things now, and they answer different questions.

The first is what the curve's *slope* is measured from, and it is thinner
than it looks: **every `game`-source item is a mistake a real human made, and
`mover_elo` says how strong they were.** That pair — an item's measurements
against the strength of the player who got it wrong — was the only external
signal in the project before launch. Everything else in the bank is engine
output, which can say what is true but not what is hard.

The second arrived with the first users and fixes what the first never could:
**the response record says how often people at a known rating actually get a
served item right**, which is a rate — the thing mining's error-only sample
has no denominator for. It is what the curve's *location* is measured from
(`rating.RESPONSE_ANCHOR`, whose comment carries the numbers), and
`trainer/fit_anchor.py` is the estimator. What it cannot do is reshape the
curve: a user's rating in the record is the model's own running estimate, so
the fit is a consistency check at the operating point — a uniform offset it
estimates well, per-item residuals it cannot see past selection. Issue #27's
IRT model is the tool for those, and this record is the data it was waiting
for.

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

**Fixing the launch-week accuracy shortfall by retargeting selection** instead
of moving the item scale (issue #82's option 1). Serves the same items, but
leaves `expected_score` mispredicting every trial by the same fourteen points
— and Elo updates on a fixed mismatch drift every rating down by about five
points per answer, forever. The model had to move, not the aim.

**Re-deriving the anchor under a chance-floored logistic link.** The
psychometrically standard 2AFC form (0.5 + 0.5·logistic) would put the anchor
near +400 rather than +131, because it disagrees with the plain logistic
about what an aggregate accuracy means away from the operating point. Both
packages match the data selection actually generates; they differ only where
nothing has been measured, and the floored one asserts a scale parameter no
fit has touched. So the plain link keeps the anchor its own measurement
produced, and the floor survives only as a bound in `expected_score` — the
part of the claim the task guarantees. `fit_anchor --link` is how the record
gets to overturn this once the offsets spread out.

## Open, and what to check when changing it

- **Which window.** `SHALLOW_PLIES` is a choice inside the noise, not an
  argmax, and its comment says so. `--windows` prints every window against two
  metrics that disagree about the answer; expect the metric you pick to matter
  more than the value you land on.
- **The top is the thin end now.** The easy end was filled on 2026-08-05 —
  every band a user from 480 to 2280 is aimed at holds over 5,000 items, and a
  beginner's holds 10,925. `trainer.supply` names what is left. The remedy is
  mining, and README's "keeping the bank full" carries the measured yields —
  the thing to know is that mining steers the deep gap while difficulty is the
  shallow one, so an order is priced per region, not per band.
  The top band is a different problem from the easy end and not fixable by
  aiming: it is every item with a negative shallow gap, so it already collects
  100% of the hardest positions the curve can name, and `HARD_CEILING` is an
  asymptote the tail approaches by halving. Deepening it is paid for in labeled
  volume at a yield of a few percent, and no user is near it — so price it
  against who it serves before ordering.
- **The responses have fixed the scale's location and nothing else.** The
  anchor is one number, measured flat over seven rating bands of launch-week
  data — re-check that flatness at ten times the sample before trusting it,
  because a non-flat anchor is a shape error no constant fixes. The link is
  still the assumed Elo logistic: nothing has fitted its 400-point base, and
  `fit_anchor --link` only becomes decisive once shared answers and
  calibration tails spread the offsets out. Difficulty stays fixed at
  labeling time on purpose (SPEC says why); issue #27 is the model that would
  feed responses all the way back — per-item, offline — and `gap_ladder`
  exists partly so that it can be refitted without re-labeling anything.
- **Answers under three seconds sit at chance** — click-through, ~5% of the
  launch-week record, cleanly separated by `response_ms` (now range-checked
  at the endpoint). The fits hold them out with `--min-ms`; whether rating
  updates should too is open, and cuts both ways — it is also an exploit for
  ducking a loss on a hard position.
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
is a property of the change, not of the app. The anchor is the instructive
special case: its user map was the identity, *because* the shift was measured
from the users' own answers — their Elo ratings were already in equilibrium
with what they could actually do, and serving them different items was the
point of the fix, not a side effect to be compensated. `meta.anchored_at`
marks the boundary for analysis; only the timestamp needed gating. Reopen the
bank with a writer before reading `trainer.supply`: it opens read-only, so it
refuses a bank still on the old scale rather than reporting bands off it.

Re-running the fit is `uv run python -m trainer.fit_difficulty`. If it stops
returning what `rating.GAP_SLOPE`'s comment claims, the estimator has drifted
and every comparison in this file is void — which is why it also prints the
same method applied to the deep gap, which is a fixed point: it returns 6096,
and if that drifts the estimator has changed rather than the data.
