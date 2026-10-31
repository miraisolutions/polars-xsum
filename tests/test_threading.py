"""The plugin's thread pool follows Polars' thread configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import polars as pl
import pytest

import polars_xsum

SCRIPT = """
import polars as pl
from polars_xsum import thread_pool_size, xsum

df = pl.DataFrame({"x": [1e16, 1.0, -1e16] * 100_000})
assert df.select(xsum("x")).item() == 100_000.0
print(thread_pool_size())
"""


def pool_size_with(env: dict[str, str]) -> int:
    """Run a fresh interpreter with ``env`` and return the plugin's pool size."""
    clean = {
        k: v for k, v in os.environ.items() if k not in {"POLARS_MAX_THREADS", "RAYON_NUM_THREADS"}
    }
    result = subprocess.run(  # noqa: S603 (trusted, fixed command)
        [sys.executable, "-c", SCRIPT],
        env=clean | env,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


@pytest.mark.skipif("RAYON_NUM_THREADS" in os.environ, reason="explicit override is set")
def test_default_matches_polars() -> None:
    assert polars_xsum.thread_pool_size() == pl.thread_pool_size()


@pytest.mark.parametrize("threads", [1, 3])
def test_follows_polars_max_threads(threads: int) -> None:
    assert pool_size_with({"POLARS_MAX_THREADS": str(threads)}) == threads


def test_rayon_num_threads_overrides() -> None:
    env = {"POLARS_MAX_THREADS": "3", "RAYON_NUM_THREADS": "2"}
    assert pool_size_with(env) == 2


ROLLING_SCRIPT = """
import numpy as np
import polars as pl
from polars_xsum import rolling_xsum, rolling_xsum_by

rng = np.random.default_rng(0)
n = 200_000
values = rng.standard_normal(n) * 10.0 ** rng.integers(-20, 20, n)
df = pl.DataFrame({"t": rng.integers(0, n, n), "x": values})
out = df.select(
    fixed=rolling_xsum("x", 1000),
    keyed=rolling_xsum_by("x", "t", "500i"),
)
print(out.select(pl.all().cast(pl.Float64).to_physical().hash()).sum().row(0))
"""


def test_rolling_sums_do_not_depend_on_threads() -> None:
    def run(threads: str) -> str:
        env = {k: v for k, v in os.environ.items() if k != "POLARS_MAX_THREADS"}
        result = subprocess.run(  # noqa: S603 (trusted, fixed command)
            [sys.executable, "-c", ROLLING_SCRIPT],
            env=env | {"RAYON_NUM_THREADS": threads},
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    assert run("1") == run("7")


GLOBAL_POOL_SCRIPT = """
import json
import os

import polars as pl
from polars_xsum import rolling_xsum, rolling_xsum_by, xsum

def threads():
    tasks = os.listdir("/proc/self/task")
    return {t: open(f"/proc/self/task/{t}/comm").read().strip() for t in tasks}

n = 200_000
x = [float(i % 1000) for i in range(n)]
df = pl.DataFrame({"x": x, "t": range(n), "g": [i % 2 for i in range(n)]})
# Start Polars' own threads and load the plugin.
df.select(naive=pl.col("x").sum(), short=xsum(pl.col("x").head(10)))
before = threads()
df.select(xsum("x"))
df.group_by("g").agg(xsum("x"))
df.select(rows=rolling_xsum("x", 100), keys=rolling_xsum_by("x", "t", "100i"))
after = threads()
print(json.dumps(sorted({name for tid, name in after.items() if tid not in before})))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="lists threads via /proc")
def test_only_the_plugin_pool_is_started() -> None:
    # Regression test: sizing the blocks of a single long sum outside the
    # plugin's pool started rayon's global pool, i.e. a second set of threads.
    result = subprocess.run(  # noqa: S603 (trusted, fixed command)
        [sys.executable, "-c", GLOBAL_POOL_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    new_threads = json.loads(result.stdout)
    assert new_threads, "the plugin's pool should have been started"
    assert all(name.startswith("xsum-") for name in new_threads), new_threads
