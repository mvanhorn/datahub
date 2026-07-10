import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import pytest

from tests.utilities import env_vars

logger = logging.getLogger(__name__)

_BARRIER_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "datahub-smoke-policy-phase"
_REMAINING_FILE = _BARRIER_DIR / "phase1-remaining"
_COMPLETE_FILE = _BARRIER_DIR / "phase1-complete"
_COUNTED_FILE = _BARRIER_DIR / "phase1-counted.txt"
_COLLECTION_INIT_FILE = _BARRIER_DIR / "collection-init"
_BARRIER_LOCK_FILE = _BARRIER_DIR / "barrier.lock"
_MUTATOR_LOCK_FILE = _BARRIER_DIR / "mutator.lock"

_PHASE1_WAIT_TIMEOUT_SECONDS = 3600.0
_PHASE1_WAIT_LOG_INTERVAL_SECONDS = 60.0

_mutator_lock_fd: Any = None
_mutator_lock_module: Optional[str] = None
_missing_barrier_logged = False


def module_has_global_policy_mutator_marker(module: Any) -> bool:
    """True when the test module declares ``pytestmark.global_policy_mutator``."""
    if module is None:
        return False
    marks = getattr(module, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    return any(getattr(mark, "name", None) == "global_policy_mutator" for mark in marks)


def _item_is_global_policy_mutator(item: pytest.Item) -> bool:
    return item.get_closest_marker("global_policy_mutator") is not None


def _is_last_test_in_module(item: Any, nextitem: Any) -> bool:
    if nextitem is None:
        return True
    item_module = getattr(item, "module", None)
    next_module = getattr(nextitem, "module", None)
    if item_module is None or next_module is None:
        return True
    return item_module.__name__ != next_module.__name__


@contextmanager
def _barrier_exclusive_lock() -> Iterator[None]:
    _BARRIER_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BARRIER_LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _phase1_barrier_status() -> Tuple[int, int]:
    with _barrier_exclusive_lock():
        if not _REMAINING_FILE.exists():
            return -1, 0
        remaining = int(_REMAINING_FILE.read_text().strip() or "0")
        counted = (
            len(_COUNTED_FILE.read_text().splitlines())
            if _COUNTED_FILE.exists() and _COUNTED_FILE.read_text().strip()
            else 0
        )
        return remaining, counted


def init_phase1_barrier(remaining: int) -> None:
    with _barrier_exclusive_lock():
        if _COMPLETE_FILE.exists():
            return
        if _REMAINING_FILE.exists():
            return

        _REMAINING_FILE.write_text(str(remaining))
        _COUNTED_FILE.write_text("")
        if remaining <= 0:
            _COMPLETE_FILE.write_text("1")


def _mark_collection_initialized() -> None:
    _BARRIER_DIR.mkdir(parents=True, exist_ok=True)
    _COLLECTION_INIT_FILE.write_text("1")


def _log_missing_barrier_once() -> None:
    global _missing_barrier_logged
    if _missing_barrier_logged:
        return
    _missing_barrier_logged = True
    worker = os.getenv("PYTEST_XDIST_WORKER", "main")
    logger.error(
        "Global policy phase 1 barrier was not initialized on worker=%s after "
        "collection; phase-1 teardown cannot decrement the shared counter",
        worker,
    )


def wait_for_phase1_complete() -> None:
    deadline = time.monotonic() + _PHASE1_WAIT_TIMEOUT_SECONDS
    next_log_at = time.monotonic()
    while time.monotonic() < deadline:
        if _COMPLETE_FILE.exists():
            return
        now = time.monotonic()
        if now >= next_log_at:
            remaining, completed = _phase1_barrier_status()
            worker = os.getenv("PYTEST_XDIST_WORKER", "main")
            logger.info(
                "Waiting for global policy phase 1 (worker=%s, remaining=%s, completed=%s)",
                worker,
                remaining,
                completed,
            )
            next_log_at = now + _PHASE1_WAIT_LOG_INTERVAL_SECONDS
        time.sleep(0.25)

    remaining, completed = _phase1_barrier_status()
    worker = os.getenv("PYTEST_XDIST_WORKER", "main")
    raise TimeoutError(
        "Timed out waiting for default global policy smoke tests to finish "
        f"(worker={worker}, remaining={remaining}, completed={completed})"
    )


def record_phase1_test_completed(nodeid: str) -> None:
    with _barrier_exclusive_lock():
        if not _REMAINING_FILE.exists():
            if _COLLECTION_INIT_FILE.exists():
                _log_missing_barrier_once()
            return
        if _COMPLETE_FILE.exists():
            return
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


def _acquire_mutator_module(module_name: str) -> None:
    global _mutator_lock_fd, _mutator_lock_module

    if _mutator_lock_module == module_name:
        return

    if _mutator_lock_module is not None:
        raise RuntimeError(
            f"Mutator lock already held by {_mutator_lock_module!r}, "
            f"cannot acquire for {module_name!r} on the same worker"
        )

    _BARRIER_DIR.mkdir(parents=True, exist_ok=True)
    lock_fd = open(_MUTATOR_LOCK_FILE, "w")
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    _mutator_lock_fd = lock_fd
    _mutator_lock_module = module_name


def _release_mutator_module() -> None:
    global _mutator_lock_fd, _mutator_lock_module

    if _mutator_lock_fd is None:
        return

    fcntl.flock(_mutator_lock_fd.fileno(), fcntl.LOCK_UN)
    _mutator_lock_fd.close()
    _mutator_lock_fd = None
    _mutator_lock_module = None


def _release_mutator_module_if_last(item: Any, nextitem: Any) -> None:
    if not _is_last_test_in_module(item, nextitem):
        return
    _release_mutator_module()


def reset_mutator_lock_state_for_tests() -> None:
    """Reset process-local mutator lock state (unit tests only)."""
    _release_mutator_module()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if env_vars.get_test_strategy() == "cypress":
        return
    if not _item_is_global_policy_mutator(item):
        return
    item_module = getattr(item, "module", None)
    if item_module is None:
        raise RuntimeError("global_policy_mutator test has no module")

    wait_for_phase1_complete()
    _acquire_mutator_module(item_module.__name__)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> None:
    if env_vars.get_test_strategy() == "cypress":
        return

    if _item_is_global_policy_mutator(item):
        _release_mutator_module_if_last(item, nextitem)
        return

    # Decrement even when the test was skipped at setup/call (e.g. skipif). Those
    # items are included in the controller's phase-1 count at collection time.
    record_phase1_test_completed(item.nodeid)


def pytest_collection_finish(session: pytest.Session) -> None:
    if session.config.getoption("collectonly"):
        return
    if env_vars.get_test_strategy() == "cypress":
        return
    # xdist workers only see their own subset in session.items; initializing here
    # would set the barrier to a partial count and race the controller.
    if hasattr(session.config, "workerinput"):
        return

    phase1_count = sum(
        1
        for item in session.items
        if not item.get_closest_marker("global_policy_mutator")
        and not item.get_closest_marker("skip")
    )
    init_phase1_barrier(phase1_count)
    _mark_collection_initialized()

    logger.info(
        "Global policy ordering: %s default-policy tests, then %s mutator tests",
        phase1_count,
        len(session.items) - phase1_count,
    )
