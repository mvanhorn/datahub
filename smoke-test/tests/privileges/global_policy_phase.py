import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.utilities import env_vars

logger = logging.getLogger(__name__)

_BARRIER_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "datahub-smoke-policy-phase"
_REMAINING_FILE = _BARRIER_DIR / "phase1-remaining"
_COMPLETE_FILE = _BARRIER_DIR / "phase1-complete"
_COUNTED_FILE = _BARRIER_DIR / "phase1-counted.txt"
_BARRIER_LOCK_FILE = _BARRIER_DIR / "barrier.lock"
_MUTATOR_LOCK_FILE = _BARRIER_DIR / "mutator.lock"

_PHASE1_WAIT_TIMEOUT_SECONDS = 3600.0


def module_has_global_policy_mutator_marker(module: Any) -> bool:
    """True when the test module declares ``pytestmark.global_policy_mutator``."""
    if module is None:
        return False
    marks = getattr(module, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    return any(getattr(mark, "name", None) == "global_policy_mutator" for mark in marks)


@contextmanager
def _barrier_exclusive_lock() -> Iterator[None]:
    _BARRIER_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BARRIER_LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def init_phase1_barrier(remaining: int) -> None:
    with _barrier_exclusive_lock():
        _REMAINING_FILE.write_text(str(remaining))
        _COUNTED_FILE.write_text("")
        if remaining <= 0:
            _COMPLETE_FILE.write_text("1")
        elif _COMPLETE_FILE.exists():
            _COMPLETE_FILE.unlink()


def wait_for_phase1_complete() -> None:
    deadline = time.monotonic() + _PHASE1_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _COMPLETE_FILE.exists():
            return
        time.sleep(0.25)
    raise TimeoutError(
        "Timed out waiting for default global policy smoke tests to finish"
    )


def record_phase1_test_completed(nodeid: str) -> None:
    with _barrier_exclusive_lock():
        counted = (
            set(_COUNTED_FILE.read_text().splitlines())
            if _COUNTED_FILE.exists() and _COUNTED_FILE.read_text().strip()
            else set()
        )
        if nodeid in counted:
            return
        counted.add(nodeid)
        _COUNTED_FILE.write_text("\n".join(sorted(counted)) + "\n")

        remaining = int(_REMAINING_FILE.read_text().strip() or "0")
        remaining -= 1
        _REMAINING_FILE.write_text(str(max(remaining, 0)))
        if remaining <= 0:
            _COMPLETE_FILE.write_text("1")
            logger.info(
                "Global policy phase 1 complete (%s tests); mutator modules may run",
                len(counted),
            )


@contextmanager
def global_policy_mutator_lock() -> Iterator[None]:
    _BARRIER_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MUTATOR_LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@pytest.fixture(scope="module", autouse=True)
def global_policy_mutator_module_gate(request):
    """After default-policy tests finish, run mutator modules one at a time."""
    module = request.module
    if not module_has_global_policy_mutator_marker(module):
        yield
        return

    wait_for_phase1_complete()
    with global_policy_mutator_lock():
        yield


def pytest_collection_finish(session: pytest.Session) -> None:
    if session.config.getoption("collectonly"):
        return
    if env_vars.get_test_strategy() == "cypress":
        return

    phase1_count = sum(
        1
        for item in session.items
        if not item.get_closest_marker("global_policy_mutator")
    )
    init_phase1_barrier(phase1_count)
    logger.info(
        "Global policy ordering: %s default-policy tests, then %s mutator tests",
        phase1_count,
        len(session.items) - phase1_count,
    )


def pytest_runtest_teardown(item: pytest.Item, nextitem) -> None:
    if item.get_closest_marker("global_policy_mutator"):
        return
    record_phase1_test_completed(item.nodeid)
