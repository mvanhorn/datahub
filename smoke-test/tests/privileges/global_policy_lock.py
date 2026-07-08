import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCK_PATH = Path(os.environ.get("TMPDIR", "/tmp")) / "datahub-smoke-global-policy.lock"


@contextmanager
def global_policy_state_lock() -> Iterator[None]:
    """Serialize policy-mutating smoke tests across pytest-xdist workers."""
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
