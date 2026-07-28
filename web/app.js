import { Chessground } from "./vendor/chessground.min.js";

const USER = new URLSearchParams(location.search).get("user") || "default";
const BRUSHES = ["blue", "yellow"]; // arrow colors matching the two buttons

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
// main board. lines[i] = {label, cls, steps: [{uci, san, fen}]}
let lines = [];
let activeLine = 0;
let stepIdx = -1; // -1 = at the decision position, before any line move
let autoplayTimer = null;
let stepMs = +(localStorage.getItem("stepMs") || 750); // auto-play pace
let lastResult = null; // /api/answer payload for the current reveal
let lastChoiceSan = null;

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
    drawable: { autoShapes: shapes },
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
    `${tag}: ${mv.san} — Stockfish eval ${mv.eval} (${mv.wp}% win probability for the side to move)\n` +
    `  Engine line: ${sans}`
  );
}

function buildCopyText() {
  const r = lastResult;
  return [
    `I'm training move discrimination in chess. Position (FEN):`,
    trial.fen,
    ``,
    `${trial.side_to_move} to move. I was asked which of these two moves is better:`,
    ``,
    describeMove(r.best, `Best move (per Stockfish)`),
    describeMove(r.distractor, `Alternative`),
    ``,
    `The gap is ${r.gap_wp}% win probability. I picked ${lastChoiceSan}, which was ${r.correct ? "correct" : "wrong"}.`,
    `Please explain in plain terms why ${r.best.san} is better and what's wrong with ${r.distractor.san} — what should I have noticed on the board?`,
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
  el("board-controls").hidden = true;
  el("speed-menu").hidden = true;
  choiceEls.forEach((b) => {
    b.disabled = false;
    b.classList.remove("picked-good", "picked-bad");
  });

  trial = await api(`/api/next?user=${encodeURIComponent(USER)}`);
  el("turn-banner").textContent = `${trial.side_to_move} to move`;
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

  choiceEls[i].classList.add(result.correct ? "picked-good" : "picked-bad");
  lastResult = result;
  lastChoiceSan = choice.san;

  // Replay lines: your pick first (it auto-plays), the other switchable.
  const mkLine = (mv, isBest, tag) => ({
    steps: mv.line,
    brush: isBest ? "green" : "red",
    cls: isBest ? "good" : "bad",
    label: `${mv.san} · ${tag}`,
  });
  lines = result.correct
    ? [mkLine(result.best, true, "your pick"), mkLine(result.distractor, false, "alternative")]
    : [mkLine(result.distractor, false, "your pick"), mkLine(result.best, true, "best move")];
  lines.forEach((l, idx) => {
    const tab = el(`tab-${idx}`);
    tab.textContent = "▶ " + l.label;
    tab.className = `line-tab ${l.cls}`;
  });

  const verdict = el("verdict");
  verdict.textContent = result.correct ? "Correct" : "Wrong";
  verdict.className = result.correct ? "good" : "bad";

  fillRow("row-best", result.best);
  fillRow("row-distractor", result.distractor);
  const source =
    result.distractor_source === "game"
      ? `the move actually played in <a href="${result.game_url}" target="_blank">the game</a>`
      : "the engine's second choice";
  el("detail").innerHTML =
    `Gap: ${result.gap_wp}% win probability. The alternative was ${source}. ` +
    `Item rating ${result.item_rating} (${result.rating_delta >= 0 ? "+" : ""}${result.rating_delta} for you).`;

  el("feedback").hidden = false;
  el("board-controls").hidden = false;
  phase = "revealed";
  activeLine = 0;
  stepIdx = -1;
  renderStep();
  autoplayFrom(0);
}

function fillRow(rowId, move) {
  const row = el(rowId);
  row.querySelector(".move").textContent = move.san;
  row.querySelector(".eval").textContent = move.eval;
  row.querySelector(".wp").textContent = `${move.wp}% win`;
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

document.addEventListener("keydown", (e) => {
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
el("ctl-gear").addEventListener("click", () => {
  el("speed-menu").hidden = !el("speed-menu").hidden;
});
document.querySelectorAll("#speed-menu button").forEach((b) => {
  if (+b.dataset.speed === stepMs) b.classList.add("active");
  b.addEventListener("click", () => {
    stepMs = +b.dataset.speed;
    localStorage.setItem("stepMs", stepMs);
    document.querySelectorAll("#speed-menu button").forEach((x) =>
      x.classList.toggle("active", x === b),
    );
    el("speed-menu").hidden = true;
  });
});

initStats();
loadTrial().catch((e) => {
  el("prompt").textContent = e.message;
});
