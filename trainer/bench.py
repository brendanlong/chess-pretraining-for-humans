"""Load benchmark for the served surface: the API and the static tree.

    uv run python -m trainer.bench           # compare against bench-baseline.json
    uv run python -m trainer.bench --save    # record what this machine does now

The job is to notice when a change makes a request slower, not to predict what
production will do. So everything that doesn't have to vary is held still:

* Fixed request *counts*, not a fixed duration. `/api/next` scans the bank
  minus what the caller has already answered, so its cost grows with the
  history a run writes. A timed run therefore measures a different mix of
  database states on a fast machine than on a slow one, and stops being a
  comparison. Counting the requests instead makes every run walk the database
  through the same states in the same order. Repetitions of the writing
  scenario are not identical — each leaves rows behind — but each starts with
  fresh guests, and per-user history is what the cost depends on, so they
  measure the same thing.
* A scratch copy of a cached template database, so a run never reads or writes
  `data/items.db` and always starts from the same rows.
* The server pinned to one core where `taskset` allows it. That is both the
  shape of the machine we deploy to (one shared vCPU) and the cheapest way to
  stop the result being a report on what else the box was doing.

The driver is several processes because one Python process talking to a
one-core server is close enough to the same speed to become the thing being
measured. Latencies are the client's view, so they include loopback and httpx;
that overhead is a constant the comparison cancels out.
"""

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage

import chess
import httpx

from . import auth, db, rating

_ROOT = Path(__file__).resolve().parent.parent
BASELINE = _ROOT / "bench-baseline.json"
# Rebuilt on demand and gitignored with the rest of `data/`. Seeding costs a
# minute of position generation and argon2 hashing, which is worth paying once.
TEMPLATE = _ROOT / "data" / "bench-template.db"
TEMPLATE_VERSION = 2  # bumped when the seeding below changes what it writes

# What the deployment serves, because `pick_item` and `unseen_count` both scan
# the bank once per request: their cost is linear in this, and a benchmark
# against a toy bank would report a fifth of the work the real one does. It is
# pinned rather than read from anywhere, since a baseline is only a comparison
# if both runs measured the same amount of work — raise it when the live bank
# grows enough to matter, and re-record.
ITEM_COUNT = 24_989
# Accounts the login scenario spends. Its per-name limit is 10 per 15 minutes,
# so this is what caps how many logins one run may measure.
LOGIN_ACCOUNTS = 192
# Accounts the read scenarios browse as, each with a plausible history behind
# it. Enough for twice the default concurrency; past this the users double up,
# which costs nothing but realism, since these scenarios only read.
WARM_ACCOUNTS = 16
WARM_HISTORY = 300
PASSWORD = "bench-password-1"

# Rate limits are part of the request path and are measured with everything
# else, but a run that exhausted one would be measuring 429s. Each virtual user
# presents its own address and moves to a fresh one every `address_block`
# requests, which keeps every counter at the depth a busy shared address would
# have without ever reaching a limit. The server is started with
# CLIENT_IP_HEADER pointed at this header, which is the same path a deployed
# instance uses to believe its proxy.
ADDRESS_HEADER = "x-bench-client"
# Repetitions per scenario, bounded because the address carries it alongside
# which scenario is running, and the two share an octet.
REPEAT_LIMIT = 8


@dataclass(frozen=True)
class Scenario:
    name: str
    doc: str
    requests: int
    # Requests one address may be charged for. Below the tightest limit the
    # scenario spends: 1200/15min for answers, 60/15min for logins by address.
    address_block: int = 500
    # Virtual users, when `-c` is the wrong number for this one in particular.
    concurrency: int | None = None

    def users(self, default: int) -> int:
        return self.concurrency or default

    @property
    def warmup(self) -> int:
        """Discarded requests, to pay for connection setup and the first read
        of every SQLite page this scenario touches."""
        return min(50, max(4, self.requests // 20))


SCENARIOS = (
    Scenario("healthz", "framework floor: routing and JSON, no database", 4000),
    Scenario("static-html", "entry point, brotli, revalidated every load", 4000),
    Scenario("static-js", "versioned bundle, brotli, served immutable", 4000),
    Scenario("static-304", "the same bundle, conditional, answered 304", 4000),
    # Requested without a digest, which is the branch `assets.py` serves
    # briefly-cached for the sake of bookmarks — and the only scenario here
    # that carries a body big enough for the copying to show.
    Scenario("static-png", "73 KB image, unversioned: body throughput", 2000),
    Scenario("next-cold", "trial for a first-time visitor with no history", 2000),
    Scenario("next-warm", f"trial for an account {WARM_HISTORY} answers in", 2000),
    Scenario("stats", "drawer numbers, including the unseen-item count", 2000),
    Scenario("trial-loop", "one step = GET /api/next then POST /api/answer", 1500),
    # Fewer requests than the rest, and exactly as many users as `auth` has
    # hash slots. Argon2 is 64 MiB and deliberately slow, so on one core this
    # is already the endpoint's ceiling; asking for more only measures the
    # 503 that `HASH_WAIT_S` exists to produce, which is a capacity fact rather
    # than a regression signal.
    Scenario(
        "login",
        "argon2, one user per hash slot",
        200,
        address_block=50,
        concurrency=auth.HASH_CONCURRENCY,
    ),
)
BY_NAME = {s.name: s for s in SCENARIOS}


# --- the template database ------------------------------------------------


def _line(rng: random.Random, board: chess.Board, first: chess.Move) -> str:
    """A principal variation: the move, then a few plies of legal continuation.

    The reveal replays this move by move, so its length is part of what an
    answer costs."""
    board = board.copy(stack=False)
    ucis = []
    for _ in range(rng.randint(4, 12)):
        ucis.append(first.uci())
        board.push(first)
        legal = list(board.legal_moves)
        if not legal:
            break
        first = rng.choice(legal)
    return " ".join(ucis)


def _cp(wp: float) -> int:
    """Centipawns that would produce this win probability, near enough. The
    server only formats it, so the inverse being approximate costs nothing."""
    return int(-400 * math.log10(1 / min(max(wp, 0.001), 0.999) - 1))


def _items(rng: random.Random, count: int):
    """Positions from random playouts, deduplicated by FEN.

    Random moves rather than fixed templates because both `chess.Board(fen)` and
    `board.san(move)` cost what the position holds, and every request on the
    trial path pays for several of each."""
    seen: set[str] = set()
    while len(seen) < count:
        board = chess.Board()
        for _ in range(rng.randint(8, 70)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            moves = list(board.legal_moves)
            fen = board.fen()
            if len(moves) < 2 or fen in seen:
                continue
            seen.add(fen)
            best, distractor = rng.sample(moves, 2)
            # Spread across the whole difficulty range, so selection has a real
            # `ORDER BY ABS(rating - target)` to do rather than a tie.
            gap = rng.uniform(0.05, 0.45)
            wp_best = rng.uniform(0.5, 0.95)
            wp_distractor = max(0.01, wp_best - gap)
            yield {
                "fen": fen,
                "best_uci": best.uci(),
                "distractor_uci": distractor.uci(),
                "distractor_source": "multipv",
                "cp_best": _cp(wp_best),
                "mate_best": None,
                "cp_distractor": _cp(wp_distractor),
                "mate_distractor": None,
                "wp_best": wp_best,
                "wp_distractor": wp_distractor,
                "gap_wp": gap,
                "pv_best": _line(rng, board, best),
                "pv_distractor": _line(rng, board, distractor),
                "learnable": 1,
                "depth_deep": 18,
                "depth_shallow": 8,
                "rating": rating.difficulty_rating(gap),
                "ply": board.ply(),
                "game_url": "https://lichess.org/bench",
                "mover_elo": 1500,
                "time_control": "300+0",
            }
            if len(seen) >= count:
                return


def build_template(path: Path, items: int) -> None:
    """Items, accounts and answer histories — everything a run starts from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = db.connect(path)
    rng = random.Random(20260801)
    print(f"seeding {items} items…", flush=True)
    conn.executemany(
        """INSERT INTO items (fen, best_uci, distractor_uci, distractor_source,
             cp_best, mate_best, cp_distractor, mate_distractor, wp_best,
             wp_distractor, gap_wp, pv_best, pv_distractor, learnable,
             depth_deep, depth_shallow, rating, ply, game_url, mover_elo,
             time_control)
           VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
             :cp_best, :mate_best, :cp_distractor, :mate_distractor, :wp_best,
             :wp_distractor, :gap_wp, :pv_best, :pv_distractor, :learnable,
             :depth_deep, :depth_shallow, :rating, :ply, :game_url, :mover_elo,
             :time_control)""",
        _items(rng, items),
    )
    # One hash for every account: they all share a password, and paying argon2
    # sixty-four times over to store sixty-four different salts of the same
    # secret would only make seeding slower. The login scenario verifies against
    # a real hash either way, which is the part that costs anything.
    print("seeding accounts…", flush=True)
    password_hash = auth.hash_password(PASSWORD)
    for i in range(LOGIN_ACCOUNTS):
        auth.create_account(
            conn,
            f"bench-login-{i}",
            password_hash,
            None,
            rating.USER_START,
            rating.CALIB_START_STEP,
        )
    for i in range(WARM_ACCOUNTS):
        user = auth.create_account(
            conn, f"bench-warm-{i}", password_hash, None, 1200.0, rating.CALIB_END_STEP - 1
        )
        # A history is what makes `pick_item`'s NOT IN subquery cost something,
        # and what a real user's requests are served against.
        conn.executemany(
            """INSERT INTO responses (user_id, item_id, choice_uci, correct,
                 response_ms, user_rating_before, user_rating_after, item_rating_before)
               VALUES (?, ?, 'e2e4', 1, 3000, 1200, 1200, 1200)""",
            [(user["id"], item_id) for item_id in range(1, WARM_HISTORY + 1)],
        )
        conn.execute("UPDATE users SET attempts = ? WHERE id = ?", (WARM_HISTORY, user["id"]))
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('bench_template', ?)",
        (f"{TEMPLATE_VERSION}:{items}",),
    )
    conn.commit()
    conn.close()


def template(items: int, reseed: bool) -> Path:
    """The template database, built if it isn't already what this run wants."""
    if not reseed and TEMPLATE.exists():
        conn = db.open_connection(TEMPLATE)
        row = conn.execute("SELECT value FROM meta WHERE key = 'bench_template'").fetchone()
        conn.close()
        if row and row["value"] == f"{TEMPLATE_VERSION}:{items}":
            return TEMPLATE
    build_template(TEMPLATE, items)
    return TEMPLATE


# --- the server under test -------------------------------------------------


def served_tree(override: Path | None) -> Path:
    """The frontend directory the server will pick, worked out the same way it
    does. Not a detail: `web-dist/` appears when someone runs `npm run build`
    and disappears when they delete it, and four scenarios transfer whichever
    one is there — so a baseline that doesn't name it can silently compare
    minified bytes against unminified ones."""
    if override is not None:
        return override
    built = _ROOT / "web-dist"
    return built if built.is_dir() else _ROOT / "web"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(db_path: Path, web_dir: Path | None, port: int, cpu: int | None):
    env = {
        **os.environ,
        "TRAINER_DB": str(db_path),
        "CLIENT_IP_HEADER": ADDRESS_HEADER,
        "PYTHONPATH": str(_ROOT),
        # Configured only to keep the "no key, using an ephemeral one" warning
        # out of the run. Nothing here outlives the process that reads it.
        "TRIAL_TOKEN_SECRET": "bench",
    }
    if web_dir is not None:
        env["TRAINER_WEB_DIR"] = str(web_dir)
    # The deployed command, minus the address it binds and the noise it logs.
    cmd = [
        sys.executable, "-m", "uvicorn", "trainer.server:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--proxy-headers", "--log-level", "warning",
    ]  # fmt: skip
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu), *cmd]
    proc = subprocess.Popen(cmd, cwd=_ROOT, env=env)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before serving")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return proc
        # Outside the handler, so a server answering something other than 200
        # waits like the rest. Spinning here would burn the core the server is
        # pinned to, which is the one thing this must not touch.
        time.sleep(0.1)
    stop_server(proc)
    raise RuntimeError("server never became healthy")


def stop_server(proc: subprocess.Popen) -> None:
    """Leave nothing holding the port or the scratch database."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # A SIGTERM it never answered. Raising here would replace whatever the
        # run actually failed on with a report about the shutdown.
        proc.kill()
        proc.wait()


# --- what each virtual user does ------------------------------------------


class VU:
    """One virtual user: a cookie jar, an address, and a request counter.

    Sequential by construction — a step starts when the last one finished — so
    the scenario's concurrency is exactly the number of these.
    """

    def __init__(
        self,
        index: int,
        count: int,
        repeat: int,
        scenario: int,
        base_url: str,
        block: int,
        ctx: dict,
    ):
        self.index = index
        self.count = count  # how many of us there are, which is the concurrency
        self.repeat = repeat
        self.scenario = scenario
        self.ctx = ctx
        self.block = block
        self.n = 0
        self.state: dict = {}
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Opened on first use, so which key a step would spend can be worked
        out — by `check_budgets`, or by a test — without opening a socket."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def headers(self) -> dict[str, str]:
        # Scenario and repetition are both part of the address. A whole run
        # happens inside one fifteen-minute window, so any two stretches of it
        # that shared an address would stack their spend into one counter —
        # and the limit they hit would be one neither of them asked for.
        # Four octets for four things that must not collide: which scenario,
        # which repetition, which user, and how far through its block it is.
        # Scenario and repetition share the first, which bounds a run at
        # `REPEAT_LIMIT` repetitions — `check_budgets` enforces that.
        return {
            ADDRESS_HEADER: f"10.{self.scenario * REPEAT_LIMIT + self.repeat}."
            f"{self.index % 256}.{(self.n // self.block) % 256}"
        }

    async def request(self, method: str, url: str, *, expect: int = 200, **kw) -> httpx.Response:
        headers = {**self.headers(), **kw.pop("headers", {})}
        r = await self.client.request(method, url, headers=headers, **kw)
        if r.status_code != expect:
            raise RuntimeError(
                f"{method} {url} -> {r.status_code} (wanted {expect}): {r.text[:200]}"
            )
        return r

    async def get(self, url: str, **kw) -> httpx.Response:
        return await self.request("GET", url, **kw)


async def _setup_none(vu: VU) -> None:
    return None


async def _setup_warm(vu: VU) -> None:
    """Adopt one of the seeded accounts by presenting its session cookie.

    Minted straight into the database rather than earned with a login, so the
    read scenarios don't spend the login limiter's budget or wait on argon2."""
    vu.client.cookies.set(auth.COOKIE_NAME, vu.ctx["warm_tokens"][vu.index % WARM_ACCOUNTS])


async def _setup_304(vu: VU) -> None:
    r = await vu.get(vu.ctx["js"])
    vu.state["etag"] = r.headers["etag"]


async def _step_healthz(vu: VU) -> None:
    await vu.get("/healthz")


async def _step_html(vu: VU) -> None:
    await vu.get("/")


async def _step_js(vu: VU) -> None:
    await vu.get(vu.ctx["js"])


async def _step_304(vu: VU) -> None:
    await vu.request("GET", vu.ctx["js"], expect=304, headers={"If-None-Match": vu.state["etag"]})


async def _step_png(vu: VU) -> None:
    await vu.get(vu.ctx["png"])


async def _step_next(vu: VU) -> None:
    await vu.get("/api/next")


async def _step_stats(vu: VU) -> None:
    await vu.get("/api/stats")


async def _step_trial(vu: VU) -> None:
    """The core loop, and the only scenario that writes. The first answer mints
    the guest row this user keeps for the rest of the run."""
    trial = (await vu.get("/api/next")).json()
    await vu.request(
        "POST",
        "/api/answer",
        json={
            "item_id": trial["item_id"],
            "trial_token": trial["trial_token"],
            "choice_uci": trial["moves"][0]["uci"],
            "response_ms": 3000,
        },
    )


# The per-name limiter allows ten guesses per fifteen minutes. A *successful*
# login clears that counter (`auth.RateLimiter.clear`, called from the login
# endpoint), and every login here succeeds, so staying under it is belt rather
# than braces — the per-address limiter, which nothing clears, is the one that
# binds. Kept anyway: it costs a modulo, and it is what makes the plan correct
# rather than accidentally correct.
LOGINS_PER_ACCOUNT = 9


def login_account(vu: VU) -> int:
    """Which seeded account this user's next attempt guesses at.

    Every user moves to a fresh account every ninth attempt, striding by the
    number of users so two never land on the same one, and each repetition
    starts where the last one left off — the per-name limit is spent for the
    window, not for the run. `check_budgets` verifies up front that it all fits
    in the accounts the template seeded.
    """
    block = vu.repeat * _login_blocks(vu.count) + vu.n // LOGINS_PER_ACCOUNT
    return (vu.index + vu.count * block) % LOGIN_ACCOUNTS


async def _step_login(vu: VU) -> None:
    account = login_account(vu)
    await vu.request(
        "POST",
        "/api/account/login",
        json={"username": f"bench-login-{account}", "password": PASSWORD},
    )


STEPS = {
    "healthz": (_setup_none, _step_healthz),
    "static-html": (_setup_none, _step_html),
    "static-js": (_setup_none, _step_js),
    "static-304": (_setup_304, _step_304),
    "static-png": (_setup_none, _step_png),
    "next-cold": (_setup_none, _step_next),
    "next-warm": (_setup_warm, _step_next),
    "stats": (_setup_warm, _step_stats),
    "trial-loop": (_setup_none, _step_trial),
    "login": (_setup_none, _step_login),
}


# --- the driver ------------------------------------------------------------


SERVER_CPU = 0


def _siblings(cpu: int) -> set[int]:
    """Every logical CPU sharing a physical core with this one.

    The driver has to stay off all of them, not just off the server's own. An
    SMT sibling shares the physical core's execution units, so a driver process
    scheduled there competes with the thing being measured for about 15% of it.
    """
    listing = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if not listing.exists():
        return {cpu}
    cpus = {cpu}
    for part in listing.read_text().strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def driver_processes(concurrency: int, cores: int, pin: bool) -> int:
    """One process per virtual user, as far as the cores allow.

    The driver has to be far enough from its own ceiling that what the table
    reports is the server. Each virtual user is a synchronous chain of awaits,
    so two sharing a process share a GIL and queue behind each other's response
    parsing — on the cheapest scenarios that alone was worth 20% of the
    throughput, which is the size of the regression this is meant to detect.
    """
    available = cores - len(_siblings(SERVER_CPU)) if pin else cores
    return max(1, min(concurrency, available))


def _pin_worker(cpus: list[int]) -> None:
    if cpus:
        os.sched_setaffinity(0, set(cpus))


async def _run_vus(plan: dict) -> tuple[list[float], float, float]:
    scenario = BY_NAME[plan["scenario"]]
    setup, step = STEPS[scenario.name]
    vus = [
        VU(
            i,
            plan["concurrency"],
            plan["repeat"],
            plan["scenario_index"],
            plan["base_url"],
            scenario.address_block,
            plan["ctx"],
        )
        for i in plan["vus"]
    ]
    for vu in vus:
        await setup(vu)

    async def drive(vu: VU, warmup: int, measured: int) -> list[float]:
        samples = []
        for i in range(warmup + measured):
            start = time.perf_counter()
            await step(vu)
            vu.n += 1
            if i >= warmup:
                samples.append((time.perf_counter() - start) * 1000)
        return samples

    # Warm up first and time only what follows, so process startup and the
    # slower first pass over every SQLite page stay out of the throughput.
    await asyncio.gather(*(drive(vu, scenario.warmup, 0) for vu in vus))
    # perf_counter, not time.time: this is the one number joined across driver
    # processes, and a wall clock can be stepped by NTP mid-run. On Linux it is
    # CLOCK_MONOTONIC, which shares an origin between processes on one machine.
    started = time.perf_counter()
    per_vu = await asyncio.gather(
        *(drive(vu, 0, n) for vu, n in zip(vus, plan["counts"], strict=True))
    )
    ended = time.perf_counter()
    for vu in vus:
        await vu.close()
    return [s for samples in per_vu for s in samples], started, ended


def _worker(plan: dict) -> tuple[list[float], float, float]:
    return asyncio.run(_run_vus(plan))


def _split(total: int, parts: int) -> list[int]:
    """As even as it goes, remainder spread over the first few."""
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def summarize(samples: list[float], elapsed: float) -> dict:
    ordered = sorted(samples)

    def pct(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)], 3)

    return {
        "steps": len(samples),
        "rps": round(len(samples) / elapsed, 1),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


def _login_blocks(users: int) -> int:
    """How many accounts one repetition of the login scenario burns per user."""
    login = BY_NAME["login"]
    return math.ceil((_split(login.requests, users)[0] + login.warmup) / LOGINS_PER_ACCOUNT)


def plan_scenario(scenario: Scenario, procs: int, plan: dict) -> list[dict]:
    """One plan per driver process, partitioning the virtual users between them.

    Requests are dealt per virtual user rather than per process, so the work is
    the same work however many processes the driver happens to get — otherwise
    the throughput reported would be a function of the machine's core count.
    """
    vus = scenario.users(plan["concurrency"])
    counts = _split(scenario.requests, vus)
    plans, assigned = [], 0
    for group in _split(vus, min(procs, vus)):
        indices = list(range(assigned, assigned + group))
        assigned += group
        plans.append(
            {
                **plan,
                "scenario": scenario.name,
                "scenario_index": SCENARIOS.index(scenario),
                "vus": indices,
                "counts": [counts[i] for i in indices],
                "concurrency": vus,
            }
        )
    return plans


def elapsed_across(results: list[tuple[list[float], float, float]]) -> float:
    """The window every driver process was inside, which is the union of theirs.

    They neither start nor stop together, so anything narrower reports a rate
    that no part of the run actually sustained.
    """
    return max(r[2] for r in results) - min(r[1] for r in results)


def run_scenario(scenario: Scenario, pool: ProcessPoolExecutor, procs: int, plan: dict) -> dict:
    """Drive one scenario to completion and rejoin the samples."""
    results = list(pool.map(_worker, plan_scenario(scenario, procs, plan)))
    samples = [s for r in results for s in r[0]]
    return summarize(samples, elapsed_across(results))


# --- reporting -------------------------------------------------------------


def fingerprint() -> dict:
    model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.M)
        model = match.group(1).strip() if match else ""
    return {
        "host": platform.node(),
        "cpu": model or platform.processor(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _delta(now: float, before: float) -> float:
    return (now - before) / before * 100 if before else 0.0


def _machine_cpu_seconds() -> float | None:
    """CPU seconds this machine has spent on anything but idling, all cores."""
    stat = Path("/proc/stat")
    if not stat.exists():
        return None
    # user nice system idle iowait irq softirq steal guest guest_nice — and
    # the last two are already counted inside user and nice, so summing the
    # line whole would bill a virtualised host's guest time twice.
    ticks = [int(t) for t in stat.read_text().split("\n", 1)[0].split()[1:10]]
    busy = sum(ticks[:3]) + sum(ticks[5:8])
    return busy / os.sysconf("SC_CLK_TCK")


def _own_cpu_seconds() -> float:
    """CPU seconds this process and its reaped children have used."""
    return sum(
        getrusage(who).ru_utime + getrusage(who).ru_stime for who in (RUSAGE_SELF, RUSAGE_CHILDREN)
    )


class Contention:
    """How much of the machine somebody else was using while we measured.

    Without this the tool cannot tell a slower app from a busier box, and the
    two look identical in the table — a run sharing the machine with a build
    reports a regression that isn't there. Our own cost is subtracted through
    `rusage`, so what's left is other people's work, in cores.
    """

    def __init__(self):
        self.machine = _machine_cpu_seconds()
        self.ours = _own_cpu_seconds()
        self.started = time.monotonic()

    def cores(self) -> float | None:
        """Only meaningful once the server and the driver pool have been
        reaped: `RUSAGE_CHILDREN` counts a child from the moment it's waited
        for, not while it runs."""
        machine = _machine_cpu_seconds()
        if machine is None or self.machine is None:
            return None
        # Both sides as deltas over the same window: `getrusage` reports since
        # this process started, which is before the window opened.
        ours = _own_cpu_seconds() - self.ours
        return max(0.0, (machine - self.machine) - ours) / (time.monotonic() - self.started)


# Other work worth naming in the output. Below this the machine is idle
# enough that the repetitions absorb it; above it, a "regression" is as likely
# to be the neighbour as the code.
BUSY_CORES = 0.5


def report(results: dict, baseline: dict | None, threshold: float, busy: float | None) -> bool:
    """Print the table; return True if anything regressed past the threshold."""
    old = (baseline or {}).get("scenarios", {})
    header = f"{'scenario':<13} {'steps/s':>9} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}"
    print("\n" + header + ("   vs baseline" if old else ""))
    print("-" * (len(header) + (14 if old else 0)))
    regressed = []
    for name, r in results.items():
        line = (
            f"{name:<13} {r['rps']:>9.1f} {r['p50_ms']:>9.2f} "
            f"{r['p95_ms']:>9.2f} {r['p99_ms']:>9.2f}"
        )
        was = old.get(name)
        if was:
            # Throughput and median together: one moving without the other is
            # usually the machine, both moving the same way is usually us.
            d_rps, d_p50 = _delta(r["rps"], was["rps"]), _delta(r["p50_ms"], was["p50_ms"])
            bad = d_rps <= -threshold or d_p50 >= threshold
            mark = "SLOWER" if bad else ("faster" if d_rps >= threshold else "")
            line += f"   {d_rps:+6.1f}% rps {d_p50:+6.1f}% p50 {mark}"
            if bad:
                regressed.append(name)
        print(line)
    if "trial-loop" in results:
        print("(a trial-loop step is two requests: the trial, then the answer)")
    if old and not regressed:
        print(f"\nno scenario regressed by more than {threshold:.0f}%.")
    elif regressed:
        print(f"\nregressed by more than {threshold:.0f}%: {', '.join(regressed)}")
    if busy is not None and busy >= BUSY_CORES:
        print(
            f"\nwarning: something else on this machine used {busy:.1f} cores "
            "while this ran, so every number above is a lower bound. Re-run on "
            "an idle machine before believing a regression."
        )
    return bool(regressed)


def compare_settings(baseline: dict, settings: dict, chosen: tuple[Scenario, ...]) -> None:
    """Refuse to call a run a comparison when it measured something else.

    Only the request counts of the scenarios actually run are checked, so
    `--only` compares its subset against the same subset of the baseline.
    """
    was_requests = baseline["settings"].get("requests", {})
    differ = {
        k: (v, settings.get(k))
        for k, v in baseline["settings"].items()
        if k != "requests" and settings.get(k) != v
    }
    differ.update(
        {
            f"{s.name} requests": (was_requests[s.name], s.requests)
            for s in chosen
            if s.name in was_requests and was_requests[s.name] != s.requests
        }
    )
    if differ:
        detail = ", ".join(f"{k}: {was} -> {now}" for k, (was, now) in differ.items())
        raise SystemExit(
            f"this run isn't comparable to the baseline ({detail}).\n"
            "Match the settings, or re-record with --save."
        )
    if baseline["machine"] != fingerprint():
        print(
            "warning: the baseline was recorded on a different machine or "
            f"interpreter ({baseline['machine'].get('host')}, "
            f"{baseline['machine'].get('cpu')}). Timings are not comparable.",
            file=sys.stderr,
        )


# --- entry point -----------------------------------------------------------


def check_budgets(chosen: tuple[Scenario, ...], concurrency: int, repeat: int) -> None:
    """Refuse a run that would spend more of a rate limit than exists.

    Cheap to check and expensive to discover halfway through: a scenario that
    trips a limiter fails on a 429 after minutes of work, and the arithmetic
    that decides is all known up front.
    """
    if not 1 <= repeat <= REPEAT_LIMIT:
        raise SystemExit(f"--repeat must be between 1 and {REPEAT_LIMIT}")
    login = next((s for s in chosen if s.name == "login"), None)
    if login is None:
        return
    users = login.users(concurrency)
    needed = repeat * users * _login_blocks(users)
    if needed > LOGIN_ACCOUNTS:
        raise SystemExit(
            f"the login scenario would need {needed} accounts at concurrency "
            f"{concurrency} over {repeat} repetitions, and the template seeds "
            f"{LOGIN_ACCOUNTS}. Raise LOGIN_ACCOUNTS (and --reseed), or lower "
            "--repeat."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Load benchmark for the API and the static tree.")
    ap.add_argument("--save", action="store_true", help="record this run as the new baseline")
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("-c", "--concurrency", type=int, default=8, help="virtual users (default 8)")
    ap.add_argument("--items", type=int, default=ITEM_COUNT, help="items in the scratch bank")
    ap.add_argument("--reseed", action="store_true", help="rebuild the template database")
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help=f"times to run each scenario, best kept (1-{REPEAT_LIMIT}, default 3)",
    )
    ap.add_argument("--threshold", type=float, default=20.0, help="regression percent (default 20)")
    ap.add_argument("--no-pin", action="store_true", help="don't confine the server to one core")
    ap.add_argument("--web-dir", type=Path, help="frontend tree to serve (default: the server's)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    args = ap.parse_args()

    chosen = SCENARIOS
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
        chosen = tuple(BY_NAME[n] for n in names)
    if args.save and args.only:
        # A baseline holding scenarios measured minutes or commits apart isn't
        # one run, and nothing downstream could tell. Re-record all of it.
        raise SystemExit("--save records the whole suite; drop --only")
    check_budgets(chosen, args.concurrency, args.repeat)

    cores = os.cpu_count() or 1
    pin = bool(not args.no_pin and shutil.which("taskset") and cores > 2)
    procs = driver_processes(args.concurrency, cores, pin)
    # Everything here changes what the numbers mean, so a run that differs in
    # any of it is not a comparison. `driver_processes` and `pinned` decide how
    # much of the machine each side gets; `web` decides what four of the
    # scenarios transfer, and it moves on its own when someone runs
    # `npm run build`.
    settings = {
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "items": args.items,
        "warm_history": WARM_HISTORY,
        "driver_processes": procs,
        "pinned": pin,
        "web": served_tree(args.web_dir).name,
        "requests": {s.name: s.requests for s in SCENARIOS},
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    if baseline and not args.save:
        compare_settings(baseline, settings, chosen)

    contention = Contention()
    source = template(args.items, args.reseed)
    scratch = _ROOT / "data" / "bench-run.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(scratch) + suffix).unlink(missing_ok=True)
    shutil.copy(source, scratch)

    port = free_port()
    server = start_server(scratch, args.web_dir, port, cpu=SERVER_CPU if pin else None)
    base_url = f"http://127.0.0.1:{port}"
    print(
        f"serving {args.items} items on {base_url} "
        f"({'pinned to cpu0' if pin else 'unpinned'}), "
        f"{args.concurrency} virtual users across {procs} driver processes"
    )
    try:
        # The digests are the server's to choose, so ask it rather than
        # rebuilding the asset tree here and hoping the two agree.
        index = httpx.get(f"{base_url}/", timeout=10).text
        js = re.search(r'src="(app\.js\?v=[0-9a-f]+)"', index)
        if not js:
            raise SystemExit("couldn't find the versioned bundle in the served index page")
        ctx = {"js": "/" + js.group(1), "png": "/social-preview.png"}
        # Sessions minted directly, which is what lets the read scenarios browse
        # as an account without spending the login limiter to get in.
        conn = db.connect(scratch)
        ctx["warm_tokens"] = [
            auth.start_session(
                conn,
                conn.execute(
                    "SELECT id FROM users WHERE name = ?", (f"bench-warm-{i}",)
                ).fetchone()["id"],
            )
            for i in range(WARM_ACCOUNTS)
        ]
        conn.commit()
        conn.close()

        plan = {"base_url": base_url, "ctx": ctx, "concurrency": args.concurrency}
        # Every core the server isn't on, so the driver can't be the thing that
        # runs out of room. Empty means leave its affinity alone.
        off_limits = _siblings(SERVER_CPU)
        driver_cpus = [c for c in range(cores) if c not in off_limits] if pin else []
        results = {}
        with ProcessPoolExecutor(procs, initializer=_pin_worker, initargs=(driver_cpus,)) as pool:
            for scenario in chosen:
                print(f"  {scenario.name}: {scenario.doc}", end="", flush=True)
                runs = [
                    run_scenario(scenario, pool, procs, {**plan, "repeat": r})
                    for r in range(args.repeat)
                ]
                # Best of the repeats, because interference only ever subtracts:
                # a slow run means something else got the core for a moment, and
                # averaging that in measures the machine rather than the code.
                # Every repeat is printed so the spread stays visible.
                best = max(runs, key=lambda r: r["rps"])
                print(" [" + ", ".join(f"{r['rps']:.0f}" for r in runs) + " req/s]", flush=True)
                results[scenario.name] = best
    finally:
        stop_server(server)

    # After the server and the pool have been reaped, so their CPU counts as
    # ours rather than as somebody else's.
    regressed = report(results, None if args.save else baseline, args.threshold, contention.cores())
    if args.save:
        args.baseline.write_text(
            json.dumps(
                {"machine": fingerprint(), "settings": settings, "scenarios": results}, indent=2
            )
            + "\n"
        )
        print(f"\nwrote {args.baseline}")
    raise SystemExit(1 if regressed else 0)


if __name__ == "__main__":
    main()
