"""Behavioral concurrency test for the sync-under-load bracket
(codex impl-r12): a shrink that commits an engine offloaded while the
sync RPC is in flight must NOT be undone by the bracket's finally —
the concurrently-shrunk engine stays out of router admission; the
still-active engine is re-registered.

The coordinator instance is built via ``object.__new__`` (bypassing the
Ray-heavy ctor) with only the fields ``sync_base_weights_to_active``
touches; every remote handle is a local fake whose ``.remote()`` returns
the plain value, and the module's ``ray.get`` is patched to unwrap.
"""

from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

import rlix.pipeline.miles_coordinator as coord_mod


def _coordinator_cls():
    cls = coord_mod.MilesCoordinator
    meta = getattr(cls, "__ray_metadata__", None)
    return meta.modified_class if meta is not None else cls


class _RM:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class _FakeEngineHandle:
    def __init__(self, log, idx):
        self.unregister_from_router = _RM(lambda: log.append(("unregister", idx)))
        self.register_with_router = _RM(lambda: log.append(("register", idx)))


class TestSyncBracketConcurrentShrink(unittest.TestCase):
    def test_finally_skips_concurrently_shrunk_engine(self):
        log: list = []
        cls = _coordinator_cls()
        coord = object.__new__(cls)
        coord._pipeline_id = "pipe-test"
        coord._resize_sync_lock = threading.Lock()
        coord._active_engine_indices = {0, 1}
        coord._cache_ready_step = None
        coord._model_update_service = object()  # _ensure returns this

        handles = {0: _FakeEngineHandle(log, 0), 1: _FakeEngineHandle(log, 1)}
        # Manager: engine 0 gets shrunk mid-sync — states reflect it.
        def _register_if_active(idx_list):
            # Mirrors the manager-side atomic method: state check and
            # register inside one serialized call.
            states = {i: ("offloaded" if i == 0 else "active") for i in idx_list}
            out = []
            for i in idx_list:
                if states[i] == "active":
                    log.append(("register", i))
                    out.append(i)
            return out

        manager = types.SimpleNamespace(
            get_engine_handles=_RM(lambda idx: dict(handles)),
            register_router_if_active=_RM(_register_if_active),
            _abort_engines=_RM(lambda idx: log.append(("abort", tuple(idx)))),
            _reset_abort_idempotency_for=_RM(
                lambda idx: log.append(("reset_idempotency", tuple(idx)))
            ),
        )
        coord._model_update_resources = {
            "cache_owner_actor": object(),
            "rollout_manager": manager,
            "train_gpu_ids": [0],
            "infer_gpu_ids": [0, 1, 2],
            "rollout_num_gpus_per_engine": 1,
            "transport_mode": "cpu_serialize",
        }

        def _sync(**kwargs):
            # Concurrent shrink lands while the sync RPC is in flight:
            # engine 0 leaves the intended-active set (its manager state
            # is already "offloaded" per the fake above).
            coord._active_engine_indices = {1}
            log.append(("sync", tuple(sorted(kwargs["target_engine_indices"]))))
            return 7

        service = types.SimpleNamespace(sync_selected_workers=_RM(_sync))
        fake_ray = types.SimpleNamespace(
            get=lambda refs: refs,  # fakes resolve eagerly; unwrap is identity
        )
        with mock.patch.object(coord_mod, "ray", fake_ray), mock.patch.object(
            cls, "_ensure_model_update_service", lambda self: service
        ):
            version = coord.sync_base_weights_to_active(7)

        self.assertEqual(version, 7)
        registered = [ev for ev in log if ev[0] == "register"]
        # Engine 1 (still active + still intended) re-registered; engine 0
        # (concurrently shrunk) must NOT be re-admitted.
        self.assertEqual(registered, [("register", 1)])
        # Quiesce ran for both before the sync.
        self.assertIn(("unregister", 0), log)
        self.assertIn(("unregister", 1), log)
        self.assertLess(log.index(("unregister", 0)), log.index(("sync", (0, 1))))
        # Idempotency reset still ran.
        self.assertTrue([ev for ev in log if ev[0] == "reset_idempotency"])

    def test_reregister_failure_after_successful_sync_escalates(self):
        # codex impl-r14: a successful sync must NOT report success when
        # the authoritative re-register raises — engines would remain
        # intended-active yet unroutable.
        log: list = []
        cls = _coordinator_cls()
        coord = object.__new__(cls)
        coord._pipeline_id = "pipe-test"
        coord._resize_sync_lock = threading.Lock()
        coord._active_engine_indices = {0, 1}
        coord._cache_ready_step = None
        handles = {0: _FakeEngineHandle(log, 0), 1: _FakeEngineHandle(log, 1)}

        def _register_boom(idx_list):
            raise RuntimeError("router add_worker 503")

        manager = types.SimpleNamespace(
            get_engine_handles=_RM(lambda idx: dict(handles)),
            register_router_if_active=_RM(_register_boom),
            _abort_engines=_RM(lambda idx: None),
            _reset_abort_idempotency_for=_RM(lambda idx: None),
        )
        coord._model_update_resources = {
            "cache_owner_actor": object(),
            "rollout_manager": manager,
            "train_gpu_ids": None,
            "infer_gpu_ids": None,
            "rollout_num_gpus_per_engine": 1,
            "transport_mode": "cpu_serialize",
        }
        service = types.SimpleNamespace(sync_selected_workers=_RM(lambda **k: 5))
        fake_ray = types.SimpleNamespace(get=lambda refs: refs)
        with mock.patch.object(coord_mod, "ray", fake_ray), mock.patch.object(
            cls, "_ensure_model_update_service", lambda self: service
        ):
            with self.assertRaisesRegex(RuntimeError, "re-registration failed"):
                coord.sync_base_weights_to_active(5)

    def test_sync_failure_keeps_original_error_over_reregister_failure(self):
        # When the sync body itself failed, that original error stays the
        # primary exception even if re-register also fails.
        log: list = []
        cls = _coordinator_cls()
        coord = object.__new__(cls)
        coord._pipeline_id = "pipe-test"
        coord._resize_sync_lock = threading.Lock()
        coord._active_engine_indices = {0}
        coord._cache_ready_step = None
        handles = {0: _FakeEngineHandle(log, 0)}
        manager = types.SimpleNamespace(
            get_engine_handles=_RM(lambda idx: dict(handles)),
            register_router_if_active=_RM(
                lambda idx: (_ for _ in ()).throw(RuntimeError("router down"))
            ),
            _abort_engines=_RM(lambda idx: None),
            _reset_abort_idempotency_for=_RM(lambda idx: None),
        )
        coord._model_update_resources = {
            "cache_owner_actor": object(),
            "rollout_manager": manager,
            "train_gpu_ids": None,
            "infer_gpu_ids": None,
            "rollout_num_gpus_per_engine": 1,
            "transport_mode": "cpu_serialize",
        }

        def _sync_fail(**k):
            raise TimeoutError("flush timeout")

        service = types.SimpleNamespace(sync_selected_workers=_RM(_sync_fail))
        fake_ray = types.SimpleNamespace(get=lambda refs: refs)
        with mock.patch.object(coord_mod, "ray", fake_ray), mock.patch.object(
            cls, "_ensure_model_update_service", lambda self: service
        ):
            with self.assertRaisesRegex(TimeoutError, "flush timeout"):
                coord.sync_base_weights_to_active(5)

    def test_finally_reregisters_all_on_no_concurrent_change(self):
        log: list = []
        cls = _coordinator_cls()
        coord = object.__new__(cls)
        coord._pipeline_id = "pipe-test"
        coord._resize_sync_lock = threading.Lock()
        coord._active_engine_indices = {0, 1}
        coord._cache_ready_step = None
        handles = {0: _FakeEngineHandle(log, 0), 1: _FakeEngineHandle(log, 1)}
        def _register_all(idx_list):
            for i in idx_list:
                log.append(("register", i))
            return list(idx_list)

        manager = types.SimpleNamespace(
            get_engine_handles=_RM(lambda idx: dict(handles)),
            register_router_if_active=_RM(_register_all),
            _abort_engines=_RM(lambda idx: None),
            _reset_abort_idempotency_for=_RM(lambda idx: None),
        )
        coord._model_update_resources = {
            "cache_owner_actor": object(),
            "rollout_manager": manager,
            "train_gpu_ids": None,
            "infer_gpu_ids": None,
            "rollout_num_gpus_per_engine": 1,
            "transport_mode": "cpu_serialize",
        }
        service = types.SimpleNamespace(
            sync_selected_workers=_RM(lambda **k: 3)
        )
        fake_ray = types.SimpleNamespace(get=lambda refs: refs)
        with mock.patch.object(coord_mod, "ray", fake_ray), mock.patch.object(
            cls, "_ensure_model_update_service", lambda self: service
        ):
            version = coord.sync_base_weights_to_active(3)
        self.assertEqual(version, 3)
        registered = sorted(ev[1] for ev in log if ev[0] == "register")
        self.assertEqual(registered, [0, 1])


if __name__ == "__main__":
    unittest.main()
