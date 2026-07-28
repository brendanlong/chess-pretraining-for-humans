import { Chessground } from "./vendor/chessground.min.js";

// User identity: URL param wins (shareable/debug), then the sticky local
// choice, then "default". Real accounts come later; see the settings drawer.
const urlUser = new URLSearchParams(location.search).get("user");
const USER = urlUser || localStorage.getItem("user") || "default";

const BRUSHES = ["blue", "purple"]; // arrow colors matching the two buttons
// Custom brush set: chessground's yellow is invisible on the light squares.
const BRUSH_DEFS = {
  blue: { key: "blue", color: "#1a56c4", opacity: 0.9, lineWidth: 10 },
  purple: { key: "purple", color: "#8a2be2", opacity: 0.9, lineWidth: 10 },
  green: { key: "green", color: "#15781b", opacity: 0.85, lineWidth: 10 },
  red: { key: "red", color: "#b02323", opacity: 0.85, lineWidth: 10 },
};

const el = (id) => document.getElementById(id);
const boardEl = el("board");
const choiceEls = [el("choice-1"), el("choice-2")];

let cg = null;
let trial = null; // current /api/next payload
let phase = "loading"; // loading | choosing | submitting | revealed
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
let lastResult = null; // /api/answer payload for the current reveal

function arrow(uci, brush) {
  return { orig: uci.slice(0, 2), dest: uci.slice(2, 4), brush };
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
    // back at the decision point: show both candidate arrows again
    setBoard(trial.fen, trial.side_to_move, [
      arrow(lines[0].steps[0].uci, lines[0].brush),
      arrow(lines[1].steps[0].uci, lines[1].brush),
    ]);
  } else {
    const step = line.steps[stepIdx];
    setBoard(step.fen, trial.side_to_move, [], {
      animate: true,
      lastMove: [step.uci.slice(0, 2), step.uci.slice(2, 4)],
    });
  }
  el("line-sans").innerHTML = line.steps
    .map((s, i) => {
      const cur = i === stepIdx ? " current" : "";
      return `<span class="ply${cur}">${s.san}</span>`;
    })
    .join(" ");
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
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function loadTrial() {
  phase = "loading";
  stopAutoplay();
  lines = [];
  stepIdx = -1;
  el("feedback").hidden = true;
  el("repeat-note").hidden = true;
  el("ask").hidden = false;
  choiceEls.forEach((b) => (b.disabled = false));

  trial = await api(`/api/next?user=${encodeURIComponent(USER)}`);
  el("turn-label").textContent = `${trial.side_to_move} to move`;
  el("turn-dot").className = trial.side_to_move;
  setBoard(
    trial.fen,
    trial.side_to_move,
    trial.moves.map((m, i) => arrow(m.uci, BRUSHES[i])),
  );
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

async function choose(i) {
  if (phase !== "choosing") return;
  phase = "submitting";
  const choice = trial.moves[i];
  choiceEls.forEach((b) => (b.disabled = true));

  const result = await api("/api/answer", {
    item_id: trial.item_id,
    choice_uci: choice.uci,
    response_ms: Math.round(performance.now() - shownAt),
    user: USER,
  });

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
    brush: isBest ? "green" : "red",
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

  const source =
    result.distractor_source === "game"
      ? `the move actually played in <a href="${result.game_url}" target="_blank">the game</a>`
      : "the engine's second choice";
  el("detail").innerHTML =
    `Gap: ${result.gap_wp}% win probability. The alternative was ${source}. ` +
    `Item rating ${result.item_rating}.`;

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
    const s = await api(`/api/stats?user=${encodeURIComponent(USER)}`);
    if (s.accuracy_last_50 != null)
      el("stat-acc").textContent = Math.round(s.accuracy_last_50 * 100) + "%";
  } catch (e) {
    console.warn(e);
  }
}

// --- settings drawer ------------------------------------------------------

function openSettings() {
  el("user-input").value = USER;
  el("settings").hidden = false;
}

function closeSettings() {
  el("settings").hidden = true;
}

function switchUser(name) {
  localStorage.setItem("user", name);
  // Drop any ?user= override so the stored name takes effect.
  location.href = location.pathname;
}

el("settings-btn").addEventListener("click", openSettings);
el("user-btn").addEventListener("click", openSettings);
el("settings-close").addEventListener("click", closeSettings);
el("settings").addEventListener("click", (e) => {
  if (e.target === el("settings")) closeSettings();
});
el("user-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = el("user-input").value.trim();
  if (name && name !== USER) switchUser(name);
  else closeSettings();
});

document.querySelectorAll("#speed-menu button").forEach((b) => {
  if (+b.dataset.speed === stepMs) b.classList.add("active");
  b.addEventListener("click", () => {
    stepMs = +b.dataset.speed;
    localStorage.setItem("stepMs", stepMs);
    document.querySelectorAll("#speed-menu button").forEach((x) =>
      x.classList.toggle("active", x === b),
    );
  });
});

// --- input ----------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (!el("settings").hidden) {
    if (e.key === "Escape") closeSettings();
    return;
  }
  if (e.target.tagName === "INPUT") return;
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
      loadTrial();
    }
  }
});
choiceEls.forEach((b, i) => b.addEventListener("click", () => choose(i)));
el("tab-0").addEventListener("click", () => switchLine(0, true));
el("tab-1").addEventListener("click", () => switchLine(1, true));
el("next").addEventListener("click", loadTrial);
el("copy-btn").addEventListener("click", copyForClaude);
el("ctl-reset").addEventListener("click", resetLine);
el("ctl-back").addEventListener("click", () => stepLine(-1));
el("ctl-fwd").addEventListener("click", () => stepLine(1));

el("user-name").textContent = USER;
initStats();
loadTrial().catch((e) => {
  el("prompt").textContent = e.message;
});
