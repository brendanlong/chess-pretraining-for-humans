import { Chessground } from "./vendor/chessground.min.js";

const USER = new URLSearchParams(location.search).get("user") || "default";
const BRUSHES = ["blue", "yellow"]; // arrow colors matching the two buttons

const el = (id) => document.getElementById(id);
const boardEl = el("board");
const choiceEls = [el("choice-1"), el("choice-2")];

let cg = null;
let trial = null; // current /api/next payload
let phase = "loading"; // loading | choosing | revealed | probe-done
let shownAt = 0;
let streak = 0;
let accWindow = []; // local last-50 correctness (feedback trials only)

function arrow(uci, brush) {
  return { orig: uci.slice(0, 2), dest: uci.slice(2, 4), brush };
}

// "~" marks a still-calibrating rating (new users start low and climb fast)
function ratingLabel(value, calibrating) {
  return (calibrating ? "~" : "") + value;
}

function setBoard(fen, orientation, shapes) {
  const config = {
    fen,
    orientation,
    viewOnly: true,
    coordinates: true,
    animation: { enabled: false },
    drawable: { autoShapes: shapes },
  };
  if (!cg) cg = Chessground(boardEl, config);
  else cg.set(config);
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
  el("feedback").hidden = true;
  el("repeat-note").hidden = true;
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
  setBoard(trial.fen, trial.side_to_move, [
    arrow(result.best.uci, "green"),
    arrow(result.distractor.uci, "red"),
  ]);

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
  phase = "revealed";
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
  if (e.key === "1" || e.key === "ArrowLeft") choose(0);
  else if (e.key === "2" || e.key === "ArrowRight") choose(1);
  else if ((e.key === " " || e.key === "Enter") && phase === "revealed") {
    e.preventDefault();
    loadTrial();
  }
});
choiceEls.forEach((b, i) => b.addEventListener("click", () => choose(i)));
el("next").addEventListener("click", loadTrial);

initStats();
loadTrial().catch((e) => {
  el("prompt").textContent = e.message;
});
