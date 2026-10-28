"""The plugin's thread pool follows Polars' thread configuration."""

from __future__ import annotations

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
