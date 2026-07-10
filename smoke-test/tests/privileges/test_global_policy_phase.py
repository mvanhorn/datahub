from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import tests.privileges.global_policy_phase as global_policy_phase


@pytest.fixture
def barrier_paths(tmp_path, monkeypatch):
    barrier_dir = tmp_path / "policy-phase"
    remaining = barrier_dir / "phase1-remaining"
    complete = barrier_dir / "phase1-complete"
    counted = barrier_dir / "phase1-counted.txt"
    collection_init = barrier_dir / "collection-init"
    barrier_lock = barrier_dir / "barrier.lock"
    mutator_lock = barrier_dir / "mutator.lock"

    monkeypatch.setattr(global_policy_phase, "_BARRIER_DIR", barrier_dir)
    monkeypatch.setattr(global_policy_phase, "_REMAINING_FILE", remaining)
    monkeypatch.setattr(global_policy_phase, "_COMPLETE_FILE", complete)
    monkeypatch.setattr(global_policy_phase, "_COUNTED_FILE", counted)
    monkeypatch.setattr(global_policy_phase, "_COLLECTION_INIT_FILE", collection_init)
    monkeypatch.setattr(global_policy_phase, "_BARRIER_LOCK_FILE", barrier_lock)
    monkeypatch.setattr(global_policy_phase, "_MUTATOR_LOCK_FILE", mutator_lock)
    monkeypatch.setattr(global_policy_phase, "_missing_barrier_logged", False)

    global_policy_phase.reset_mutator_lock_state_for_tests()

    return SimpleNamespace(
        remaining=remaining,
        complete=complete,
        counted=counted,
        collection_init=collection_init,
    )


class TestModuleHasGlobalPolicyMutatorMarker:
    def test_true_when_marker_present(self) -> None:
        module = SimpleNamespace(
            pytestmark=[pytest.mark.global_policy_mutator],
        )
        assert global_policy_phase.module_has_global_policy_mutator_marker(module)

    def test_false_when_marker_absent(self) -> None:
        module = SimpleNamespace(pytestmark=[])
        assert not global_policy_phase.module_has_global_policy_mutator_marker(module)


class TestInitPhase1Barrier:
    def test_initializes_remaining_and_complete_when_zero(self, barrier_paths) -> None:
        global_policy_phase.init_phase1_barrier(0)

        assert barrier_paths.remaining.read_text() == "0"
        assert barrier_paths.complete.exists()

    def test_second_init_does_not_reset_progress(self, barrier_paths) -> None:
        global_policy_phase.init_phase1_barrier(2)
        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")

        global_policy_phase.init_phase1_barrier(99)

        assert barrier_paths.remaining.read_text() == "1"
        assert not barrier_paths.complete.exists()
        assert "tests/a.py::test_one" in barrier_paths.counted.read_text()


class TestRecordPhase1TestCompleted:
    def test_no_op_when_barrier_not_initialized(self, barrier_paths) -> None:
        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")

        assert not barrier_paths.remaining.exists()
        assert not barrier_paths.complete.exists()

    def test_logs_when_collection_init_without_barrier(
        self, barrier_paths, caplog
    ) -> None:
        barrier_paths.collection_init.parent.mkdir(parents=True, exist_ok=True)
        barrier_paths.collection_init.write_text("1")

        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")

        assert not barrier_paths.remaining.exists()
        assert any(
            "barrier was not initialized" in record.message for record in caplog.records
        )

    def test_decrements_remaining_and_marks_complete(self, barrier_paths) -> None:
        global_policy_phase.init_phase1_barrier(2)

        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")
        assert barrier_paths.remaining.read_text() == "1"
        assert not barrier_paths.complete.exists()

        global_policy_phase.record_phase1_test_completed("tests/b.py::test_two")
        assert barrier_paths.remaining.read_text() == "0"
        assert barrier_paths.complete.exists()

    def test_duplicate_nodeid_is_no_op(self, barrier_paths) -> None:
        global_policy_phase.init_phase1_barrier(1)

        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")
        global_policy_phase.record_phase1_test_completed("tests/a.py::test_one")

        assert barrier_paths.remaining.read_text() == "0"
        assert barrier_paths.complete.exists()


class TestMutatorModuleLock:
    def test_acquire_and_release_module_lock(self, barrier_paths) -> None:
        global_policy_phase._acquire_mutator_module("tests.privileges.test_privileges")
        assert global_policy_phase._mutator_lock_module == (
            "tests.privileges.test_privileges"
        )
        assert global_policy_phase._mutator_lock_fd is not None

        global_policy_phase._acquire_mutator_module("tests.privileges.test_privileges")
        global_policy_phase._release_mutator_module()

        assert global_policy_phase._mutator_lock_module is None
        assert global_policy_phase._mutator_lock_fd is None

    def test_is_last_test_in_module(self) -> None:
        mod_a = SimpleNamespace(__name__="mod_a")
        mod_b = SimpleNamespace(__name__="mod_b")
        item_a1 = SimpleNamespace(module=mod_a)
        item_a2 = SimpleNamespace(module=mod_a)
        item_b = SimpleNamespace(module=mod_b)

        assert global_policy_phase._is_last_test_in_module(item_a1, None)
        assert global_policy_phase._is_last_test_in_module(item_a1, item_b)
        assert not global_policy_phase._is_last_test_in_module(item_a1, item_a2)


def _make_collection_item(
    *,
    is_mutator: bool = False,
    is_skip: bool = False,
) -> SimpleNamespace:
    def get_closest_marker(name: str) -> object | None:
        if name == "global_policy_mutator" and is_mutator:
            return pytest.mark.global_policy_mutator
        if name == "skip" and is_skip:
            return pytest.mark.skip
        return None

    return SimpleNamespace(get_closest_marker=get_closest_marker)


def _make_phase1_teardown_item(
    nodeid: str,
    *,
    skipped_at_setup: bool = False,
    skipped_at_call: bool = False,
) -> SimpleNamespace:
    store: dict[str, object] = {}
    if skipped_at_setup:
        store["rep_setup"] = SimpleNamespace(skipped=True)
    if skipped_at_call:
        store["rep_call"] = SimpleNamespace(skipped=True)
    return SimpleNamespace(
        nodeid=nodeid,
        get_closest_marker=lambda name: None,
        _store=store,
    )


class TestPytestCollectionFinishBarrierInit:
    def test_xdist_worker_does_not_initialize_barrier(
        self, barrier_paths, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            global_policy_phase.env_vars, "get_test_strategy", lambda: "pytests"
        )
        session = SimpleNamespace(
            items=[_make_collection_item()] * 5,
            config=SimpleNamespace(
                getoption=lambda _name: False,
                workerinput={"workerid": "gw0"},
            ),
        )

        global_policy_phase.pytest_collection_finish(cast(pytest.Session, session))

        assert not barrier_paths.remaining.exists()
        assert not barrier_paths.collection_init.exists()

    def test_controller_initializes_barrier_with_full_batch_count(
        self, barrier_paths, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            global_policy_phase.env_vars, "get_test_strategy", lambda: "pytests"
        )
        session = SimpleNamespace(
            items=[
                _make_collection_item(),
                _make_collection_item(),
                _make_collection_item(is_mutator=True),
                _make_collection_item(is_skip=True),
            ],
            config=SimpleNamespace(getoption=lambda _name: False),
        )

        global_policy_phase.pytest_collection_finish(cast(pytest.Session, session))

        assert barrier_paths.remaining.read_text() == "2"
        assert barrier_paths.collection_init.exists()
        assert not barrier_paths.complete.exists()


class TestPhase1TeardownHook:
    def test_teardown_decrements_when_skipped_at_setup(
        self, barrier_paths, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            global_policy_phase.env_vars, "get_test_strategy", lambda: "pytests"
        )
        global_policy_phase.init_phase1_barrier(1)
        item = _make_phase1_teardown_item(
            "tests/a.py::test_skipif",
            skipped_at_setup=True,
        )

        global_policy_phase.pytest_runtest_teardown(cast(pytest.Item, item), None)

        assert barrier_paths.remaining.read_text() == "0"
        assert barrier_paths.complete.exists()

    def test_teardown_decrements_when_skipped_at_call(
        self, barrier_paths, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            global_policy_phase.env_vars, "get_test_strategy", lambda: "pytests"
        )
        global_policy_phase.init_phase1_barrier(1)
        item = _make_phase1_teardown_item(
            "tests/a.py::test_skipif",
            skipped_at_call=True,
        )

        global_policy_phase.pytest_runtest_teardown(cast(pytest.Item, item), None)

        assert barrier_paths.remaining.read_text() == "0"
        assert barrier_paths.complete.exists()
