"""What the benchmark's arithmetic has to get right before it runs anything.

The measuring is left to a real run — none of these start a server. They cover
the parts a bad run would only reveal minutes in, or worse, not reveal at all:
a scenario that trips a rate limit, and a seeded bank the API can't serve.
"""

from collections import Counter

import chess
import pytest

from trainer import bench


def spend(scenario, users: int, repeats: int):
    """Replay a scenario's whole plan, counting what it charges to each key.

    The same split `main` uses, so what this counts is what a run would spend.
    """
    counts = bench._split(scenario.requests, users)
    accounts: Counter[int] = Counter()
    addresses: Counter[str] = Counter()
    for repeat in range(repeats):
        for i in range(users):
            vu = bench.VU(i, users, repeat, "http://x", scenario.address_block, {})
            for _ in range(scenario.warmup + counts[i]):
                addresses[vu.headers()[bench.ADDRESS_HEADER]] += 1
                if scenario.name == "login":
                    accounts[bench.login_account(vu)] += 1
                vu.n += 1
    return accounts, addresses


def test_the_login_plan_stays_under_the_per_name_limit():
    """The limiter allows ten guesses at one name per fifteen minutes, and a
    whole run happens well inside one window — repetitions included."""
    login = bench.BY_NAME["login"]
    accounts, _ = spend(login, login.users(8), repeats=3)
    assert accounts, "the login scenario spent nothing"
    assert max(accounts.values()) <= bench.LOGINS_PER_ACCOUNT
    assert max(accounts) < bench.LOGIN_ACCOUNTS


@pytest.mark.parametrize("scenario", bench.SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("users", [1, 4, 8, 16])
def test_no_address_is_charged_past_its_block(scenario, users):
    """Rotation is what keeps `answer_limiter` (1200 per address per fifteen
    minutes) and the login counters out of the measurement."""
    _, addresses = spend(scenario, scenario.users(users), repeats=3)
    assert max(addresses.values()) <= scenario.address_block


def test_a_run_that_could_not_fit_is_refused_before_it_starts():
    login = bench.BY_NAME["login"]
    bench.check_budgets((login,), concurrency=8, repeat=3)  # the default, and it fits
    with pytest.raises(SystemExit, match="accounts"):
        bench.check_budgets((login,), concurrency=8, repeat=100)


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


def _result(rps: float, p50: float) -> dict:
    return {"steps": 100, "rps": rps, "p50_ms": p50, "p95_ms": p50, "p99_ms": p50, "mean_ms": p50}


def test_a_slower_run_is_reported_as_one():
    baseline = {"scenarios": {"healthz": _result(1000, 1.0)}}
    assert bench.report({"healthz": _result(700, 1.4)}, baseline, threshold=20.0)
    assert not bench.report({"healthz": _result(950, 1.05)}, baseline, threshold=20.0)
    # Nothing to compare against is not a regression.
    assert not bench.report({"healthz": _result(1, 1000.0)}, None, threshold=20.0)


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
