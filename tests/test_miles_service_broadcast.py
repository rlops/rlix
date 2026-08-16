"""Unit tests for MilesModelUpdateService broadcast unlock (rlops/rlix#42).

Covers plan miles-nccl-broadcast v7 evidence items:

- E2: service no longer raises for a non-empty broadcast set; comm_ranks
  is a per-GPU rank_offset cursor with uniform stride from the injected
  ``rollout_num_gpus_per_engine``; ``world_size = 1 + per_engine ×
  len(broadcast_set)``; per_engine == 1 degenerates to the dense case.
- E6(a): the SharedStorage port claim is released only AFTER the
  ``run_sync_session`` ref resolves (teardown ack), on both the success
  and the sender-raised-exception paths.
- E6(b): the wedged-sender path (service session deadline fires while the
  sender ref is unresolved) with a broadcast leg LEAKS the claim
  (no delete dispatched); a cpu_serialize-only session keeps the
  historical release-on-timeout.

Ray is imported for real (tests run on the GPU instance) but no cluster
is started: the ``@ray.remote`` wrapper is bypassed via
``__ray_metadata__.modified_class`` and every handle is a local fake with
awaitable refs.
"""

from __future__ import annotations

import asyncio
import types
import unittest
from unittest import mock

import rlix.pipeline.miles_model_update_service as svc_mod


def _service_cls():
    cls = svc_mod.MilesModelUpdateService
    meta = getattr(cls, "__ray_metadata__", None)
    return meta.modified_class if meta is not None else cls


class _Ref:
    """Awaitable standing in for a Ray ObjectRef."""

    def __init__(self, value=None, exc=None, event_log=None, event=None):
        self._value = value
        self._exc = exc
        self._event_log = event_log
        self._event = event

    def __await__(self):
        async def _resolve():
            if self._event_log is not None and self._event is not None:
                self._event_log.append(self._event)
            if self._exc is not None:
                raise self._exc
            return self._value

        return _resolve().__await__()


class _PendingRef:
    """Never-resolving awaitable (wedged sender)."""

    def __await__(self):
        async def _forever():
            await asyncio.Future()

        return _forever().__await__()


class _RM:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeEngine:
    def __init__(self):
        self.pause_generation = _RM(lambda **k: _Ref(None))
        self.finalize_weight_update = _RM(lambda **k: _Ref(None))
        self.continue_generation = _RM(lambda **k: _Ref(None))


class _FakeCacheOwner:
    def __init__(self, log, sync_ref_factory):
        self.plans = []
        self._log = log

        def _run(plan):
            self.plans.append(plan)
            return sync_ref_factory()

        # Mirrors miles RayActor.get_free_port: deterministic scan from a
        # base, honoring start_port (the collision-retry contract).
        self.get_node_ip = _RM(lambda: _Ref("10.0.0.1"))
        self.get_free_port = _RM(lambda start_port=29500, **_k: _Ref(int(start_port)))
        self.run_sync_session = _RM(_run)


class _FakeManager:
    def __init__(self, engine_indices):
        self._engines = {i: _FakeEngine() for i in engine_indices}
        self.get_engine_handles = _RM(lambda idx: _Ref(dict(self._engines)))
        self.set_weight_version = _RM(lambda *a, **k: _Ref(None))


class _FakeSharedStorage:
    def __init__(self, log, taken_keys=()):
        self._log = log
        self._taken = set(taken_keys)
        self.try_put = _RM(self._try_put)
        self.delete = _RM(self._delete)

    def _try_put(self, key, owner):
        if key in self._taken:
            self._log.append(("claim_collision", key))
            return _Ref(False)
        self._log.append(("claim_put", key))
        return _Ref(True)

    def _delete(self, key):
        self._log.append(("claim_delete", key))
        return _Ref(True)


class _Harness:
    def __init__(self, *, engine_indices, per_engine=1, sync_ref_factory=None):
        self.log: list = []
        factory = sync_ref_factory or (
            lambda: _Ref(0, event_log=self.log, event=("sender_resolved",))
        )
        self.cache_owner = _FakeCacheOwner(self.log, factory)
        self.manager = _FakeManager(engine_indices)
        self.storage = _FakeSharedStorage(self.log)
        self.cancelled: list = []
        cls = _service_cls()
        self.svc = cls(
            pipeline_id="pipe-test",
            cache_owner_actor=self.cache_owner,
            rollout_manager=self.manager,
            rollout_num_gpus_per_engine=per_engine,
        )

    def run(self, *, target, broadcast=None, timeout_s=None):
        if timeout_s is not None:
            self.svc._timeout_s = timeout_s
        fake_ray = types.SimpleNamespace(
            cancel=lambda ref, force=False: self.cancelled.append(ref)
        )
        with mock.patch.object(svc_mod, "ray", fake_ray), mock.patch.object(
            svc_mod, "_get_shared_storage_actor", lambda: self.storage
        ):
            return asyncio.run(
                self.svc.sync_selected_workers(
                    sync_id="sync-test",
                    target_engine_indices=target,
                    version=7,
                    broadcast_local_ranks=broadcast,
                )
            )

    def events(self, name):
        return [ev for ev in self.log if ev[0] == name]


class TestE2RankCursor(unittest.TestCase):
    def test_broadcast_set_no_longer_raises_and_cursor_per_engine_2(self):
        h = _Harness(engine_indices={0, 1, 2}, per_engine=2)
        version = h.run(target={0, 1, 2}, broadcast={0, 1})
        self.assertEqual(version, 7)
        self.assertEqual(len(h.cache_owner.plans), 1)
        plan = h.cache_owner.plans[0]
        # Cursor with stride 2: engine 0 -> rank_offset 1, engine 1 -> 3.
        self.assertEqual(plan["comm_ranks"][0], 1)
        self.assertEqual(plan["comm_ranks"][1], 3)
        self.assertEqual(plan["comm_ranks"][2], 0)  # cpu_serialize placeholder
        # world_size = 1 sender + 2 engines x 2 GPUs.
        self.assertEqual(plan["world_size"], 5)
        self.assertEqual(sorted(plan["broadcast_local_ranks"]), [0, 1])
        self.assertEqual(sorted(plan["cpu_serialize_local_ranks"]), [2])

    def test_per_engine_1_degenerates_to_dense_ranks(self):
        h = _Harness(engine_indices={0, 1}, per_engine=1)
        h.run(target={0, 1}, broadcast={0, 1})
        plan = h.cache_owner.plans[0]
        self.assertEqual(plan["comm_ranks"], {0: 1, 1: 2})
        self.assertEqual(plan["world_size"], 3)

    def test_empty_broadcast_keeps_world_size_1(self):
        h = _Harness(engine_indices={0, 1}, per_engine=4)
        h.run(target={0, 1})
        plan = h.cache_owner.plans[0]
        self.assertEqual(plan["world_size"], 1)
        self.assertEqual(sorted(plan["cpu_serialize_local_ranks"]), [0, 1])

    def test_broadcast_must_be_subset_of_target(self):
        h = _Harness(engine_indices={0, 1}, per_engine=1)
        with self.assertRaises(ValueError):
            h.run(target={0}, broadcast={0, 5})

    def test_invalid_per_engine_rejected_at_ctor(self):
        with self.assertRaises(ValueError):
            _service_cls()(
                pipeline_id="p",
                cache_owner_actor=object(),
                rollout_manager=object(),
                rollout_num_gpus_per_engine=0,
            )


class TestPortClaimCollision(unittest.TestCase):
    def test_contested_port_repicks_past_the_claim(self):
        # rlix#42 concurrent-sync fix: a peer pipeline holds the claim on
        # the deterministic first pick (e.g. 29500). The retry must ask
        # get_free_port to scan PAST the contested port — the port is
        # only claim-reserved, not OS-bound, so a plain rescan would
        # return the same value forever and exhaust the retry budget.
        h = _Harness(engine_indices={0}, per_engine=1)
        h.storage = _FakeSharedStorage(
            h.log, taken_keys={"MASTER_ADDR_PORT:10.0.0.1:29500"}
        )
        h.run(target={0}, broadcast={0})
        plan = h.cache_owner.plans[0]
        self.assertEqual(plan["master_port"], 29501)
        self.assertEqual(len(h.events("claim_collision")), 1)
        self.assertEqual(len(h.events("claim_put")), 1)

    def test_unexpected_storage_lookup_error_fails_fast(self):
        # codex impl-r7: a transient/unexpected claim-store lookup error
        # must fail the sync, not silently skip claims (unprotected) or
        # fall back to a second store (split claim namespace).
        h = _Harness(engine_indices={0}, per_engine=1)

        def _boom():
            raise RuntimeError("transient storage lookup failure")

        fake_ray = types.SimpleNamespace(cancel=lambda ref, force=False: None)
        with mock.patch.object(svc_mod, "ray", fake_ray), mock.patch.object(
            svc_mod, "_get_shared_storage_actor", _boom
        ):
            with self.assertRaisesRegex(RuntimeError, "transient storage"):
                asyncio.run(
                    h.svc.sync_selected_workers(
                        sync_id="s",
                        target_engine_indices={0},
                        version=1,
                        broadcast_local_ranks={0},
                    )
                )
        # Nothing reached the sender: fail-fast happened at plan build.
        self.assertEqual(h.cache_owner.plans, [])

    def test_port_claim_store_semantics(self):
        cls = svc_mod._PortClaimStore
        meta = getattr(cls, "__ray_metadata__", None)
        store = (meta.modified_class if meta is not None else cls)()
        self.assertTrue(store.try_put("k1", "p1"))
        self.assertFalse(store.try_put("k1", "p2"))  # second claimer loses
        self.assertTrue(store.delete("k1"))
        self.assertTrue(store.try_put("k1", "p2"))  # free after delete
        self.assertTrue(store.delete("missing"))  # idempotent


class TestE6PortClaimOwnership(unittest.TestCase):
    def test_claim_released_only_after_sender_resolves_success(self):
        h = _Harness(engine_indices={0, 1}, per_engine=1)
        h.run(target={0, 1}, broadcast={0})
        resolved = h.log.index(("sender_resolved",))
        deletes = [i for i, ev in enumerate(h.log) if ev[0] == "claim_delete"]
        self.assertEqual(len(deletes), 1, f"log: {h.log}")
        self.assertLess(resolved, deletes[0])

    def test_claim_released_when_sender_resolves_with_exception(self):
        h = _Harness(
            engine_indices={0},
            per_engine=1,
            sync_ref_factory=lambda: _Ref(exc=RuntimeError("sender aborted")),
        )
        h.log  # sync_ref_factory closes over nothing that logs resolution
        with self.assertRaisesRegex(RuntimeError, "sender aborted"):
            h.run(target={0}, broadcast={0})
        # Sender ref resolved (with an error) => teardown ack => release.
        self.assertEqual(len(h.events("claim_delete")), 1)

    def test_wedged_sender_with_broadcast_leaks_claim(self):
        h = _Harness(
            engine_indices={0},
            per_engine=1,
            sync_ref_factory=lambda: _PendingRef(),
        )
        with self.assertRaises(asyncio.TimeoutError):
            h.run(target={0}, broadcast={0}, timeout_s=0.2)
        # C6 v4 rule: no delete dispatched — claim intentionally leaked.
        self.assertEqual(h.events("claim_delete"), [])
        # Inflight refs were cancelled.
        self.assertTrue(h.cancelled)

    def test_wedged_sender_cpu_serialize_only_still_releases(self):
        h = _Harness(
            engine_indices={0},
            per_engine=1,
            sync_ref_factory=lambda: _PendingRef(),
        )
        with self.assertRaises(asyncio.TimeoutError):
            h.run(target={0}, timeout_s=0.2)
        # Historical behavior preserved: no TCP store => release on abort.
        self.assertEqual(len(h.events("claim_delete")), 1)

    def test_cancel_during_post_resolution_release_still_releases(self):
        # codex impl-r1 medium: sender ref resolves, but the service
        # deadline lands during the claim-release await. The claim must
        # be released via the fire-and-forget path (the TCP store is
        # already retired), NOT leaked as a wedged sender.
        h = _Harness(engine_indices={0}, per_engine=1)
        # First delete dispatch pends forever (release await never
        # completes); the fire-and-forget retry must dispatch a second
        # delete after the timeout.
        deletes = []

        def _delete(key):
            deletes.append(key)
            if len(deletes) == 1:
                h.log.append(("claim_delete", key))
                return _PendingRef()
            h.log.append(("claim_delete", key))
            return _Ref(True)

        h.storage.delete = _RM(_delete)
        with self.assertRaises(asyncio.TimeoutError):
            h.run(target={0}, broadcast={0}, timeout_s=0.3)
        # Sender resolved before the wedge, so no leak: two delete
        # dispatches (the pending awaited one + the nowait retry).
        self.assertEqual(len(h.events("claim_delete")), 2)


if __name__ == "__main__":
    unittest.main()
