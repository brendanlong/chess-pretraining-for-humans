"""What the benchmark's arithmetic has to get right before it runs anything.

The measuring is left to a real run — none of these start a server. They cover
the parts a bad run would only reveal minutes in, or worse, not reveal at all:
a scenario that trips a rate limit, and a seeded bank the API can't serve.
"""

from collections import Counter

import chess
import pytest

from trainer import bench, db
from trainer.rating import difficulty_rating, shallow_gap_of


def spend(scenario, users: int, repeats: int, accounts=None, addresses=None):
    """Replay a scenario's whole plan, counting what it charges to each key.

    The same split `main` uses, so what this counts is what a run would spend.
    The counters can be passed in, because the limiters a real run meets are
    shared between scenarios and only a tally across all of them says whether
    the plan fits.
    """
    counts = bench._split(scenario.requests, users)
    accounts = Counter() if accounts is None else accounts
    addresses = Counter() if addresses is None else addresses
    index = bench.SCENARIOS.index(scenario)
    for repeat in range(repeats):
        for i in range(users):
            vu = bench.VU(i, users, repeat, index, "http://x", scenario.address_block, {})
            for _ in range(scenario.warmup + counts[i]):
                addresses[vu.headers()[bench.ADDRESS_HEADER]] += 1
                if scenario.name == "login":
                    accounts[bench.login_account(vu)] += 1
                vu.n += 1
    return accounts, addresses


@pytest.mark.parametrize("users", [1, 2, 4, 8, 16, 31])
@pytest.mark.parametrize("repeats", [1, 3, bench.REPEAT_LIMIT])
def test_the_login_plan_stays_under_the_per_name_limit(users, repeats):
    """Ten guesses at one name per fifteen minutes, and a whole run happens
    inside one window. The account index wraps on `LOGIN_ACCOUNTS`, so running
    out shows up as two users quietly sharing a name rather than as an error —
    which is what `check_budgets` is for, and what this checks it caught."""
    login = bench.BY_NAME["login"]
    seats = login.users(users)
    try:
        bench.check_budgets((login,), users, repeats)
    except SystemExit:
        return  # refused up front, which is the other correct answer
    accounts, _ = spend(login, seats, repeats)
    assert accounts, "the login scenario spent nothing"
    assert max(accounts.values()) <= bench.LOGINS_PER_ACCOUNT


@pytest.mark.parametrize("users", [1, 4, 8, 16])
def test_no_address_is_charged_past_its_block(users):
    """Rotation is what keeps `answer_limiter` (1200 per address per fifteen
    minutes) and the login counters out of the measurement.

    Tallied over every scenario at once, against one set of counters, because
    that is how the server meets them: one run, one fifteen-minute window, one
    limiter per key. Two scenarios that happened to present the same address
    would stack into a limit neither of them asked for.
    """
    addresses: Counter[str] = Counter()
    for scenario in bench.SCENARIOS:
        spend(scenario, scenario.users(users), repeats=3, addresses=addresses)
    worst = max(addresses.values())
    assert worst <= max(s.address_block for s in bench.SCENARIOS)
    for scenario in bench.SCENARIOS:
        _, own = spend(scenario, scenario.users(users), repeats=3)
        assert max(own.values()) <= scenario.address_block, scenario.name


def test_a_run_that_could_not_fit_is_refused_before_it_starts():
    login = bench.BY_NAME["login"]
    bench.check_budgets((login,), concurrency=8, repeat=3)  # the default, and it fits
    with pytest.raises(SystemExit, match="between 1 and"):
        bench.check_budgets((login,), concurrency=8, repeat=100)


@pytest.mark.parametrize("procs", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("scenario", bench.SCENARIOS, ids=lambda s: s.name)
def test_every_request_is_dealt_to_exactly_one_process(scenario, procs):
    """However the driver is split, the work is the same work."""
    plans = bench.plan_scenario(scenario, procs, {"concurrency": 8})
    seats = scenario.users(8)
    assert sorted(i for p in plans for i in p["vus"]) == list(range(seats))
    assert sum(sum(p["counts"]) for p in plans) == scenario.requests
    assert all(p["counts"] for p in plans), "a process with nothing to do"


def test_the_seeded_bank_fits_the_schema_it_will_be_written_to(tmp_path):
    """The seeder spells out the `items` columns, so it is a second copy of a
    schema that moves. Insert into the real one, then read back what the server
    would read: a column added or dropped fails here rather than four minutes
    into a run, and a rating the seeder computed differently from `db.connect`
    would have the first repetition paying to rewrite the whole bank."""
    conn = db.connect(tmp_path / "seeded.db")
    bench.build_template(tmp_path / "seeded.db", items=12)
    conn = db.connect(tmp_path / "seeded.db")
    rows = conn.execute("SELECT * FROM items").fetchall()
    assert len(rows) == 12
    for row in rows:
        assert row["learnable"] == 1
        assert row["rating"] == difficulty_rating(row["shallow_gap"])
        assert shallow_gap_of(row["gap_ladder"]) == row["shallow_gap"]
    # Nothing left for `db.connect` to re-derive, which is what would otherwise
    # rewrite every row the first time the server opened the file.
    assert not conn.execute(
        "SELECT 1 FROM items WHERE rating != difficulty_rating(shallow_gap) LIMIT 1"
    ).fetchone()


def test_seeded_items_are_positions_the_server_can_serve():
    """`/api/next` renders both moves as SAN and the reveal replays the whole
    line, so an illegal move in the bank is a 500 in the middle of a run."""
    for item in bench._items(bench.random.Random(0), 40):
        board = chess.Board(item["fen"])
        best = chess.Move.from_uci(item["best_uci"])
        distractor = chess.Move.from_uci(item["distractor_uci"])
        assert best in board.legal_moves and distractor in board.legal_moves
        assert best != distractor
        for uci, line in (("best", item["pv_best"]), ("distractor", item["pv_distractor"])):
            replay = chess.Board(item["fen"])
            for move in line.split():
                assert chess.Move.from_uci(move) in replay.legal_moves, (uci, line)
                replay.push_uci(move)


def test_percentiles_come_off_the_samples_not_a_model():
    summary = bench.summarize([float(n) for n in range(1, 101)], elapsed=2.0)
    assert summary["steps"] == 100
    assert summary["rps"] == 50.0
    assert (summary["p50_ms"], summary["p95_ms"], summary["p99_ms"]) == (50.0, 95.0, 99.0)


def test_requests_are_dealt_out_whole():
    assert sum(bench._split(1500, 8)) == 1500
    assert max(bench._split(1500, 8)) - min(bench._split(1500, 8)) <= 1


def test_throughput_spans_every_driver_process():
    """`rps` is steps over wall time, and the processes neither start nor stop
    together — so the window has to be the union, or a run reports a rate no
    part of it ever sustained."""
    results = [([1.0, 2.0], 100.0, 104.0), ([3.0], 101.0, 106.0)]
    assert bench.elapsed_across(results) == 6.0  # 106 - 100, not 104 - 100


def _result(rps: float, p50: float, cpu_ms: float | None = None) -> dict:
    return {
        "steps": 100,
        "rps": rps,
        "p50_ms": p50,
        "p95_ms": p50,
        "p99_ms": p50,
        "mean_ms": p50,
        "cpu_ms": cpu_ms,
    }


def test_a_slower_run_is_reported_as_one():
    baseline = {"scenarios": {"healthz": _result(1000, 1.0)}}
    slower = {"healthz": _result(700, 1.4)}
    assert bench.report(slower, baseline, threshold=20.0, busy=0.0, busy_limit=1.2)
    assert not bench.report(
        {"healthz": _result(950, 1.05)}, baseline, threshold=20.0, busy=0.0, busy_limit=1.2
    )
    # Nothing to compare against is not a regression.
    assert not bench.report(
        {"healthz": _result(1, 1000.0)}, None, threshold=20.0, busy=0.0, busy_limit=1.2
    )


def test_the_verdict_follows_the_metric_that_survives_a_shared_machine(capsys):
    """Throughput is worth failing over only when the cores were ours. Cpu per
    step is worth failing over either way: measured under full load it moved
    12-14% where throughput moved 56-70%, so it is the one that still means
    something when a neighbour is on the box."""
    baseline = {"scenarios": {"stats": _result(1000, 1.0, cpu_ms=0.5)}}
    same_work = {"stats": _result(700, 1.4, cpu_ms=0.51)}
    more_work = {"stats": _result(700, 1.4, cpu_ms=0.67)}

    # Quiet machine: losing a third of the throughput is a regression whatever
    # the cpu did.
    assert bench.report(same_work, baseline, threshold=20.0, busy=0.0, busy_limit=1.2)
    # Busy machine, same work per step: reported, not failed.
    assert not bench.report(same_work, baseline, threshold=20.0, busy=4.0, busy_limit=1.2)
    assert "(busy)" in capsys.readouterr().out
    # Busy machine, a third more cpu for the same step: under the widened bar a
    # contended run is held to, because that much is what contention alone was
    # measured doing to the write path.
    assert not bench.report(more_work, baseline, threshold=20.0, busy=4.0, busy_limit=1.2)
    # Twice the work, though, is nobody's neighbour.
    doubled = {"stats": _result(700, 1.4, cpu_ms=1.0)}
    assert bench.report(doubled, baseline, threshold=20.0, busy=4.0, busy_limit=1.2)


def test_a_busy_machine_is_disclosed(capsys):
    """A run that shared the machine reports a regression it can't stand behind,
    so it has to say which it was."""
    baseline = {"scenarios": {"healthz": _result(1000, 1.0)}}
    bench.report({"healthz": _result(700, 1.4)}, baseline, threshold=20.0, busy=4.0, busy_limit=1.2)
    # Beside the table, not on stderr: it is the line saying the table can't be
    # trusted, and a log that captured only stdout would keep one without it.
    assert "4.0 of the server's cores" in capsys.readouterr().out
    bench.report({"healthz": _result(700, 1.4)}, baseline, threshold=20.0, busy=0.1, busy_limit=1.2)
    assert "cores" not in capsys.readouterr().out


def test_a_subset_run_compares_against_the_same_subset():
    """`--only` measures fewer scenarios, which is not the same thing as
    measuring different ones."""
    settings = {"concurrency": 8, "repeat": 3, "items": 10, "warm_history": 300, "requests": {}}
    baseline = {
        "machine": bench.fingerprint(),
        "settings": {**settings, "requests": {"healthz": 4000, "login": 200}},
    }
    healthz = bench.Scenario("healthz", "", 4000)
    bench.compare_settings(baseline, settings, (healthz,))
    with pytest.raises(SystemExit, match="requests"):
        bench.compare_settings(baseline, settings, (bench.Scenario("healthz", "", 10),))
    with pytest.raises(SystemExit, match="concurrency"):
        bench.compare_settings(baseline, {**settings, "concurrency": 4}, (healthz,))


def test_rows_near_the_drivers_ceiling_are_marked(capsys):
    """A scenario the harness can barely outrun reports a change smaller than
    it is, so the row has to say which kind of row it is."""
    results = {"healthz": _result(3000, 2.4), "trial-loop": _result(170, 45.0)}
    bench.report(results, None, threshold=20.0, busy=0.0, busy_limit=1.2, ceiling=10_000.0)
    out = capsys.readouterr().out
    assert "healthz" in out.split("~ within")[1]
    assert "trial-loop" not in out.split("~ within")[1]
    # No calibration means no claim either way.
    bench.report(results, None, threshold=20.0, busy=0.0, busy_limit=1.2, ceiling=None)
    assert "~ within" not in capsys.readouterr().out
