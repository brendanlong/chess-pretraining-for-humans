import { Chessground } from "./vendor/chessground.min.js";

// Identity lives entirely in an HttpOnly session cookie, minted by the first
// answer — there is nothing to type and nothing here to spoof.
let account = { username: null, guest: true };

const BRUSHES = ["choice-1", "choice-2"]; // arrow colors matching the two buttons
// Custom brush set: chessground's own colors include a yellow that is
// invisible on the light squares, and its blue and purple are a pair most
// colorblind viewers can't tell apart. The stylesheet owns the palette (see
// its :root comment) so an arrow can never disagree with its button.
// The fallbacks are only reached if the stylesheet failed to load, where an
// empty color would make chessground draw no arrow at all.
const ROOT_STYLE = getComputedStyle(document.documentElement);
const BRUSH_DEFS = Object.fromEntries(
  [
    ["choice-1", "--arrow-1", "#1550c8"],
    ["choice-2", "--arrow-2", "#b3520a"],
    ["best", "--arrow-good", "#15781b"],
    ["worse", "--arrow-bad", "#b3117a"],
  ].map(([key, varName, fallback]) => [
    key,
    {
      key,
      color: ROOT_STYLE.getPropertyValue(varName).trim() || fallback,
      opacity: 0.8,
      lineWidth: 10,
    },
  ]),
);

const el = (id) => document.getElementById(id);
const boardEl = el("board");
const choiceEls = [el("choice-1"), el("choice-2")];
const PROMPT_HTML = el("prompt").innerHTML;

let cg = null;
let trial = null; // current /api/next payload
let pendingTrial = null; // prefetched /api/next promise, or null
let phase = "loading"; // loading | choosing | submitting | revealed | error
let shownAt = 0;
let streak = 0;
let accWindow = []; // local last-50 correctness (feedback trials only)

// Reveal replay state: two engine lines, one active, stepped through on the
// main board. lines[i] = {mv, tag, cls, brush, steps}
let lines = [];
let activeLine = 0;
let stepIdx = -1; // -1 = at the decision position, before any line move
let autoplayTimer = null;
let stepMs = +(localStorage.getItem("stepMs") || 750); // auto-play pace
// Numbered arrows are the default: they cost a reader nothing, and without
// them the pairing is back to colour alone. Off is for anyone who reads the
// colours fine and would rather have an unmarked board.
let arrowNumbers = localStorage.getItem("arrowNumbers") !== "off";
let lastResult = null; // /api/answer payload for the current reveal

// `num` is drawn as a numbered disc near the arrowhead, in the arrow's own
// colour, matching the badge on the panel control the arrow belongs to. Colour
// alone can't carry that pairing: the two arrows often cross or share a square,
// and a viewer who can't separate the hues has nothing else to go on.
function arrow(uci, brush, num) {
  const shape = { orig: uci.slice(0, 2), dest: uci.slice(2, 4), brush };
  if (arrowNumbers) shape.label = { text: String(num) };
  return shape;
}

// The badges belong to the arrows, so they come and go with them: with numbers
// off the buttons keep their small keyboard hint and nothing else.
function applyArrowNumbers() {
  document.body.classList.toggle("no-arrow-numbers", !arrowNumbers);
}

const candidateArrows = () => trial.moves.map((m, i) => arrow(m.uci, BRUSHES[i], i + 1));

// Redraw whatever the board is currently showing. Only the settings drawer
// needs this — every other change of what's on the board goes through the
// call that changed it.
function redrawBoard() {
  if (phase === "choosing" || phase === "submitting")
    setBoard(trial.fen, trial.side_to_move, candidateArrows());
  else if (phase === "revealed") renderStep();
}

// The scheme is checked rather than trusted: a `javascript:` href runs on click
// even when assigned as a property instead of parsed from markup. A URL we
// can't vouch for becomes text, which is the honest way to render a link we
// won't follow.
function gameLink(url) {
  let href = null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "https:" || parsed.protocol === "http:") href = parsed.href;
  } catch {
    // not an absolute URL at all (older items store "")
  }
  if (!href) return "the game it came from";
  const a = document.createElement("a");
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = "the game";
  return a;
}

// "~" marks a still-calibrating rating (new users start low and climb fast)
function ratingLabel(value, calibrating) {
  return (calibrating ? "~" : "") + value;
}

function setBoard(fen, orientation, shapes, opts = {}) {
  const config = {
    fen,
    orientation,
    viewOnly: true,
    coordinates: true,
    animation: { enabled: opts.animate ?? false, duration: 250 },
    lastMove: opts.lastMove,
    drawable: { autoShapes: shapes, brushes: BRUSH_DEFS },
  };
  if (!cg) cg = Chessground(boardEl, config);
  else cg.set(config);
}

// --- reveal replay ------------------------------------------------------

function stopAutoplay() {
  if (autoplayTimer) clearTimeout(autoplayTimer);
  autoplayTimer = null;
}

function renderStep() {
  const line = lines[activeLine];
  if (stepIdx < 0) {
    // Back at the decision point: show both candidate arrows again. They are
    // numbered by line, not by the button they were picked from, so the discs
    // agree with the line cards and with what keys 1 and 2 now do.
    setBoard(
      trial.fen,
      trial.side_to_move,
      lines.map((l, i) => arrow(l.steps[0].uci, l.brush, i + 1)),
    );
  } else {
    const step = line.steps[stepIdx];
    setBoard(step.fen, trial.side_to_move, [], {
      animate: true,
      lastMove: [step.uci.slice(0, 2), step.uci.slice(2, 4)],
    });
  }
  // Nodes rather than markup. SAN comes from python-chess and can't contain an
  // HTML metacharacter, so this isn't a hole — but every innerHTML fed by
  // server data is one more thing to have to think about.
  el("line-sans").replaceChildren(
    ...line.steps.flatMap((s, i) => {
      const span = document.createElement("span");
      span.className = i === stepIdx ? "ply current" : "ply";
      span.textContent = s.san;
      return i ? [" ", span] : [span];
    }),
  );
  lines.forEach((l, i) => {
    el(`tab-${i}`).classList.toggle("active", i === activeLine);
  });
}

function stepLine(delta) {
  if (phase !== "revealed") return;
  stopAutoplay();
  const max = lines[activeLine].steps.length - 1;
  stepIdx = Math.max(-1, Math.min(max, stepIdx + delta));
  renderStep();
}

function switchLine(i, autoplay) {
  if (phase !== "revealed" || !lines[i]) return;
  stopAutoplay();
  activeLine = i;
  stepIdx = -1;
  renderStep();
  if (autoplay) autoplayFrom(0);
}

function autoplayFrom(idx) {
  stopAutoplay();
  const play = () => {
    if (phase !== "revealed") return;
    if (stepIdx >= lines[activeLine].steps.length - 1) return;
    stepIdx += 1;
    renderStep();
    autoplayTimer = setTimeout(play, stepMs);
  };
  stepIdx = idx - 1;
  autoplayTimer = setTimeout(play, Math.min(stepMs, 650));
}

function resetLine() {
  if (phase !== "revealed") return;
  stopAutoplay();
  stepIdx = -1;
  renderStep();
}

// --- copy-for-Claude ----------------------------------------------------

function describeMove(mv, tag) {
  const sans = mv.line.map((s) => s.san).join(" ");
  return (
    `${mv.san}${tag} — Stockfish eval ${mv.eval}, ${mv.wp}% win probability for the side to move\n` +
    `  Engine line: ${sans}`
  );
}

// Just the facts (position, moves, evals, lines) with no question attached,
// so the user can ask their own — "I thought Bd3 was better because…".
function buildCopyText() {
  const r = lastResult;
  return [
    `Chess position (FEN):`,
    trial.fen,
    ``,
    `${trial.side_to_move} to move. Two candidate moves, evaluated by Stockfish:`,
    ``,
    describeMove(r.best, ` (the engine's best move)`),
    describeMove(r.distractor, ``),
    ``,
    `The gap between the moves is ${r.gap_wp}% win probability.`,
    ``,
  ].join("\n");
}

async function copyForClaude() {
  if (!lastResult) return;
  const text = buildCopyText();
  window.__lastCopyText = text; // debugging/testing hook
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // http (non-secure context, e.g. over the tailnet): legacy fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  const btn = el("copy-btn");
  const old = btn.innerHTML;
  btn.textContent = "Copied ✓";
  setTimeout(() => (btn.innerHTML = old), 1500);
}

async function api(path, body) {
  const res = await fetch(path, body && {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    // FastAPI puts human-readable auth failures in {"detail": "..."}.
    let detail = null;
    try {
      detail = JSON.parse(text).detail;
    } catch {
      // not JSON; fall through to the raw text
    }
    const err = new Error(typeof detail === "string" ? detail : `${path}: ${res.status} ${text}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// The reveal is the only idle stretch in the loop, and the next trial is a
// round trip the server answers in a few milliseconds — so spend it there
// rather than after the user asks for it. `/api/next` writes nothing, so a
// prefetch nobody redeems costs a query and is forgotten; seen-ness is defined
// by the `responses` row an answer writes, never by having been offered.
//
// Only ever called after an answer, which is what mints an identity, so the
// token is bound to a real user — and the one identity change that doesn't
// reload the page (signup) claims that same row, so it stays redeemable.
// Failure resolves to null instead of rejecting: nothing is awaiting this yet,
// and a prefetch that couldn't run is not an error the user should ever see.
function prefetchTrial() {
  pendingTrial = api("/api/next").catch(() => null);
}

async function loadTrial() {
  phase = "loading";
  stopAutoplay();
  lines = [];
  stepIdx = -1;
  el("feedback").hidden = true;
  el("repeat-note").hidden = true;
  el("ask").hidden = false;
  el("prompt").innerHTML = PROMPT_HTML;
  choiceEls.forEach((b) => (b.disabled = false));

  // Take the prefetch if one landed; a failed one falls back to a live fetch
  // here, where an error does have somewhere to go.
  const prefetched = pendingTrial;
  pendingTrial = null;
  trial = (prefetched && (await prefetched)) || (await api("/api/next"));
  el("turn-label").textContent = `${trial.side_to_move} to move`;
  el("turn-dot").className = trial.side_to_move;
  setBoard(trial.fen, trial.side_to_move, candidateArrows());
  trial.moves.forEach((m, i) => {
    choiceEls[i].querySelector(".san").textContent = m.san;
  });
  el("stat-rating").textContent = ratingLabel(trial.user_rating, trial.calibrating);
  el("stat-trial").textContent = trial.trial_number;
  el("stat-remaining").textContent = trial.items_remaining;
  if (trial.repeat) el("repeat-note").hidden = false;
  phase = "choosing";
  shownAt = performance.now();
}

// A failed /api/next leaves no trial to interact with: park in an error
// state that the prompt (tap) or space/enter can retry from.
function showLoadError(e) {
  phase = "error";
  el("feedback").hidden = true;
  el("ask").hidden = false;
  choiceEls.forEach((b) => (b.disabled = true));
  el("prompt").textContent = `${e.message} — tap here to retry`;
}

function nextTrial() {
  loadTrial().catch(showLoadError);
}

async function choose(i) {
  if (phase !== "choosing") return;
  phase = "submitting";
  const choice = trial.moves[i];
  choiceEls.forEach((b) => (b.disabled = true));

  let result;
  try {
    result = await api("/api/answer", {
      item_id: trial.item_id,
      // The server's own proof that it offered us this item. Answering is also
      // what mints the identity, so on a first visit this is the request that
      // gets us a cookie — nothing before it wrote anything.
      trial_token: trial.trial_token,
      choice_uci: choice.uci,
      response_ms: Math.round(performance.now() - shownAt),
    });
  } catch (err) {
    if (err.status === 409) {
      // Our trial token is no longer redeemable: it expired, or the session it
      // was issued to changed under us. Retrying the same pick would fail
      // forever, so fetch a trial this session can actually answer.
      el("prompt").textContent = "That trial has expired — loading a fresh one…";
      // Whatever made this token unredeemable applies to a prefetched one too,
      // so drop it rather than walking into the same 409 on the next answer.
      pendingTrial = null;
      nextTrial();
      return;
    }
    // Submit failed: back to choosing so the same pick can be retried.
    phase = "choosing";
    choiceEls.forEach((b) => (b.disabled = false));
    el("prompt").textContent = `${err.message} — pick again to retry`;
    return;
  }

  // Before the reveal is built, so the request overlaps rendering it. It has to
  // be after the answer landed, not before: the trial it picks is selected
  // against the rating this answer just moved, and it carries the trial number
  // and remaining count that go with it.
  prefetchTrial();

  el("stat-rating").textContent = ratingLabel(result.user_rating, result.calibrating);

  streak = result.correct ? streak + 1 : 0;
  if (!result.repeat) {
    accWindow.push(result.correct ? 1 : 0);
    if (accWindow.length > 50) accWindow.shift();
  }
  el("stat-streak").textContent = streak;
  if (accWindow.length)
    el("stat-acc").textContent =
      Math.round((100 * accWindow.reduce((a, b) => a + b, 0)) / accWindow.length) + "%";

  lastResult = result;

  // Replay lines: your pick first (it auto-plays), the other switchable.
  const mkLine = (mv, isBest, tag) => ({
    mv,
    tag,
    steps: mv.line,
    brush: isBest ? "best" : "worse",
    cls: isBest ? "good" : "bad",
  });
  lines = result.correct
    ? [mkLine(result.best, true, "your pick"), mkLine(result.distractor, false, "alternative")]
    : [mkLine(result.distractor, false, "your pick"), mkLine(result.best, true, "best move")];
  lines.forEach((l, idx) => {
    const card = el(`tab-${idx}`);
    card.classList.remove("good", "bad", "active");
    card.classList.add(l.cls);
    card.querySelector(".san").textContent = l.mv.san;
    card.querySelector(".tag").textContent = l.tag;
    card.querySelector(".eval").textContent = l.mv.eval;
    card.querySelector(".wp").textContent = `${l.mv.wp}% win`;
  });

  const verdict = el("verdict");
  verdict.textContent = result.correct ? "✓ Correct" : "✗ Wrong";
  verdict.className = result.correct ? "good" : "bad";
  el("rating-delta").textContent = result.repeat
    ? "rerun — not rated"
    : `${result.rating_delta >= 0 ? "+" : ""}${result.rating_delta} Elo`;

  // Built as nodes, not markup. `game_url` is the `Site` header of a mined PGN
  // — the one string here that didn't originate in this codebase — and
  // interpolating it into innerHTML would make a hostile PGN a stored XSS.
  const detail = el("detail");
  detail.replaceChildren(`Gap: ${result.gap_wp}% win probability. The alternative was `);
  if (result.distractor_source === "game") {
    detail.append("the move actually played in ");
    detail.append(gameLink(result.game_url));
  } else {
    detail.append("the engine's second choice");
  }
  detail.append(`. Item rating ${result.item_rating}.`);

  el("ask").hidden = true;
  el("feedback").hidden = false;
  phase = "revealed";
  activeLine = 0;
  stepIdx = -1;
  renderStep();
  autoplayFrom(0);
}

async function initStats() {
  try {
    const s = await api("/api/stats");
    if (s.account) account = s.account;
    if (s.accuracy_last_50 != null)
      el("stat-acc").textContent = Math.round(s.accuracy_last_50 * 100) + "%";
  } catch (e) {
    // Fall through to setAccount anyway: the drawer's account sections start
    // hidden in the HTML, so the guest view (and its sign-in forms) has to be
    // applied even when stats can't be fetched.
    console.warn(e);
  }
  setAccount(account);
}

// --- account --------------------------------------------------------------

function setAccount(a) {
  account = a;
  el("user-name").textContent = a.guest ? "Guest" : a.username;
  el("user-btn").title = a.guest ? "Sign up or sign in" : `Signed in as ${a.username}`;
  el("account-guest").hidden = !a.guest;
  el("account-user").hidden = a.guest;
  if (!a.guest) el("account-name").textContent = a.username;
  showDeleteConfirm(false); // a just-claimed account shouldn't open on this
}

// Deletion is irreversible, so it takes two deliberate steps: reveal the
// form, then re-enter the password the server will check anyway.
function showDeleteConfirm(open) {
  showAuthError(null);
  el("delete-form").hidden = !open;
  el("delete-btn").hidden = open;
  el("delete-password").value = "";
  if (open) el("delete-password").focus();
}

function showAuthError(message) {
  const box = el("auth-error");
  box.textContent = message || "";
  box.hidden = !message;
}

function showAuthForm(which) {
  showAuthError(null);
  el("signup-form").hidden = which !== "signup";
  el("login-form").hidden = which !== "login";
  el("tab-signup").classList.toggle("active", which === "signup");
  el("tab-login").classList.toggle("active", which === "login");
}

// Guards double submits and gives the button something to say while argon2
// works (deliberately slow, ~100ms+).
async function submitAuth(btn, run) {
  if (btn.disabled) return;
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "…";
  showAuthError(null);
  try {
    await run();
  } catch (e) {
    showAuthError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// --- settings drawer ------------------------------------------------------

let settingsReturnFocus = null;

function openSettings(focusAccount) {
  settingsReturnFocus = document.activeElement;
  showDeleteConfirm(false); // also clears any error left from last time
  el("settings").hidden = false;
  const target = focusAccount && account.guest ? el("tab-signup") : el("settings-close");
  target.focus();
}

function closeSettings() {
  el("settings").hidden = true;
  if (settingsReturnFocus instanceof HTMLElement) settingsReturnFocus.focus();
  settingsReturnFocus = null;
}

el("settings-btn").addEventListener("click", () => openSettings(false));
el("user-btn").addEventListener("click", () => openSettings(true));
el("settings-close").addEventListener("click", closeSettings);
el("settings").addEventListener("click", (e) => {
  if (e.target === el("settings")) closeSettings();
});

el("tab-signup").addEventListener("click", () => showAuthForm("signup"));
el("tab-login").addEventListener("click", () => showAuthForm("login"));

el("signup-form").addEventListener("submit", (e) => {
  e.preventDefault();
  // Signing up claims the guest row this session has been playing on, so
  // there is nothing to reload: same user, now with a name.
  submitAuth(e.submitter ?? el("signup-form").querySelector("button"), async () => {
    setAccount(
      await api("/api/account/signup", {
        username: el("signup-username").value,
        password: el("signup-password").value,
        email: el("signup-email").value || null,
      }),
    );
    el("signup-password").value = "";
  });
});

el("login-form").addEventListener("submit", (e) => {
  e.preventDefault();
  submitAuth(e.submitter ?? el("login-form").querySelector("button"), async () => {
    await api("/api/account/login", {
      username: el("login-username").value,
      password: el("login-password").value,
    });
    // Different user, so rating, stats and the in-flight trial are all stale.
    location.reload();
  });
});

el("logout-btn").addEventListener("click", (e) => {
  submitAuth(e.currentTarget, async () => {
    await api("/api/account/logout", {});
    location.reload();
  });
});

el("delete-btn").addEventListener("click", () => showDeleteConfirm(true));
el("delete-cancel").addEventListener("click", () => showDeleteConfirm(false));

el("delete-form").addEventListener("submit", (e) => {
  e.preventDefault();
  submitAuth(e.submitter ?? el("delete-form").querySelector("button"), async () => {
    // Cancel has to go down with the submit button: the request is already
    // away, so collapsing the form back to its un-armed state would tell the
    // user they'd called it off moments before the account disappears.
    el("delete-cancel").disabled = true;
    try {
      await api("/api/account/delete", { password: el("delete-password").value });
    } finally {
      el("delete-cancel").disabled = false;
    }
    // The row and its cookie are both gone; reloading picks up a fresh guest
    // rather than leaving a signed-in header over nothing.
    location.reload();
  });
});

// A settings row where one of the buttons is the current value: mark it,
// remember the choice, and hand it to whoever cares.
function segmented(menuId, storageKey, current, onPick) {
  const buttons = [...el(menuId).querySelectorAll("button")];
  const mark = (v) => buttons.forEach((b) => b.classList.toggle("active", b.dataset.value === v));
  mark(current);
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      localStorage.setItem(storageKey, b.dataset.value);
      mark(b.dataset.value);
      onPick(b.dataset.value);
    }),
  );
}

segmented("speed-menu", "stepMs", String(stepMs), (v) => {
  stepMs = +v;
});

segmented("numbers-menu", "arrowNumbers", arrowNumbers ? "on" : "off", (v) => {
  arrowNumbers = v === "on";
  applyArrowNumbers();
  redrawBoard();
});

// --- input ----------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (!el("settings").hidden) {
    if (e.key === "Escape") closeSettings();
    return;
  }
  if (phase === "choosing") {
    if (e.key === "1" || e.key === "ArrowLeft") choose(0);
    else if (e.key === "2" || e.key === "ArrowRight") choose(1);
  } else if (phase === "revealed") {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepLine(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepLine(1);
    } else if (e.key === "1") switchLine(0, true);
    else if (e.key === "2") switchLine(1, true);
    else if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      nextTrial();
    }
  } else if (phase === "error" && (e.key === " " || e.key === "Enter")) {
    e.preventDefault();
    nextTrial();
  }
});
choiceEls.forEach((b, i) => b.addEventListener("click", () => choose(i)));
el("tab-0").addEventListener("click", () => switchLine(0, true));
el("tab-1").addEventListener("click", () => switchLine(1, true));
el("next").addEventListener("click", nextTrial);
el("copy-btn").addEventListener("click", copyForClaude);
el("ctl-reset").addEventListener("click", resetLine);
el("ctl-back").addEventListener("click", () => stepLine(-1));
el("ctl-fwd").addEventListener("click", () => stepLine(1));
el("prompt").addEventListener("click", () => {
  if (phase === "error") nextTrial();
});

// Nothing at boot writes anything — identity is minted by the first answer —
// so these are safe to race. /api/stats carries the account for the header;
// if it fails, the page keeps its default guest view and the drawer's forms
// still work.
applyArrowNumbers();
initStats();
nextTrial();
