"""Unit tests for the rlix#42 transport-mode wiring (plan v7, M3).

- E3: mode resolution (default/unset → cpu_serialize; invalid env →
  fail-fast) and classification per mode: `cpu_serialize` never
  classifies; `auto` splits overlap→cpu_serialize / disjoint→broadcast
  (the only mixing mode); strict `broadcast` raises on any colocate
  target and never mixes.
- E10: v7 uniformity startup guard — resolved SGLang config server-group
  ``num_gpus_per_engine`` overrides diverging from
  ``rollout_num_gpus_per_engine`` fail-fast naming the offender.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from rlix.pipeline.miles_coordinator import (
    _ASSERT_MIXED_ENV,
    _TRANSPORT_MODE_ENV,
    _assert_mixed_transport_startup,
    _classify_broadcast_engines,
    _resolve_transport_mode,
)
from rlix.pipeline.miles_pipeline import _assert_uniform_engine_gpu_counts


class TestE3ModeResolution(unittest.TestCase):
    def test_unset_env_defaults_to_cpu_serialize(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(_TRANSPORT_MODE_ENV, None)
            self.assertEqual(_resolve_transport_mode(), "cpu_serialize")

    def test_valid_modes_parse(self):
        for mode in ("cpu_serialize", "broadcast", "auto"):
            with mock.patch.dict("os.environ", {_TRANSPORT_MODE_ENV: mode}):
                self.assertEqual(_resolve_transport_mode(), mode)

    def test_invalid_mode_fails_fast(self):
        with mock.patch.dict("os.environ", {_TRANSPORT_MODE_ENV: "brodcast"}):
            with self.assertRaisesRegex(RuntimeError, "not a valid transport mode"):
                _resolve_transport_mode()


class TestE3Classification(unittest.TestCase):
    # The M4 run (a) harness topology (run_smoke_dual.sh defaults).
    P1 = dict(train_gpu_ids=[0], infer_gpu_ids=[0, 1, 2], per_engine=1)
    P2 = dict(train_gpu_ids=[3], infer_gpu_ids=[1, 2, 3], per_engine=1)

    def test_cpu_serialize_mode_never_classifies(self):
        out = _classify_broadcast_engines(
            target_engine_indices={0, 1, 2},
            mode="cpu_serialize",
            train_gpu_ids=None,
            infer_gpu_ids=None,
            per_engine=1,
        )
        self.assertEqual(out, frozenset())

    def test_auto_mixes_on_overlap_topology_p1(self):
        out = _classify_broadcast_engines(
            target_engine_indices={0, 1, 2}, mode="auto", **self.P1
        )
        # e0@gpu0 shares the train pool -> cpu_serialize; e1@1, e2@2 -> broadcast.
        self.assertEqual(out, frozenset({1, 2}))

    def test_auto_mixes_on_overlap_topology_p2(self):
        out = _classify_broadcast_engines(
            target_engine_indices={0, 1, 2}, mode="auto", **self.P2
        )
        # infer pool sorted [1,2,3]: e0@1, e1@2 -> broadcast; e2@3 -> colocate.
        self.assertEqual(out, frozenset({0, 1}))

    def test_strict_broadcast_rejects_colocate_target(self):
        with self.assertRaisesRegex(RuntimeError, r"colocate engines \[0\]"):
            _classify_broadcast_engines(
                target_engine_indices={0, 1, 2}, mode="broadcast", **self.P1
            )

    def test_strict_broadcast_all_disjoint_passes(self):
        out = _classify_broadcast_engines(
            target_engine_indices={0, 1},
            mode="broadcast",
            train_gpu_ids=[0],
            infer_gpu_ids=[1, 2],
            per_engine=1,
        )
        self.assertEqual(out, frozenset({0, 1}))

    def test_strict_broadcast_subset_target_avoiding_colocate_passes(self):
        # Same overlap topology, but the sync target set excludes the
        # colocate engine — strict mode serves it.
        out = _classify_broadcast_engines(
            target_engine_indices={1, 2}, mode="broadcast", **self.P1
        )
        self.assertEqual(out, frozenset({1, 2}))

    def test_non_default_mode_requires_topology(self):
        with self.assertRaisesRegex(RuntimeError, "requires topology"):
            _classify_broadcast_engines(
                target_engine_indices={0},
                mode="auto",
                train_gpu_ids=None,
                infer_gpu_ids=None,
                per_engine=1,
            )

    def test_engine_outside_pool_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "maps outside the infer pool"):
            _classify_broadcast_engines(
                target_engine_indices={5}, mode="auto", **self.P1
            )

    def test_per_engine_2_uses_gpu_slices(self):
        out = _classify_broadcast_engines(
            target_engine_indices={0, 1},
            mode="auto",
            train_gpu_ids=[0],
            infer_gpu_ids=[0, 1, 2, 3],
            per_engine=2,
        )
        # e0 -> gpus [0,1] (touches train gpu 0) -> colocate; e1 -> [2,3].
        self.assertEqual(out, frozenset({1}))

    def test_empty_target_returns_empty(self):
        out = _classify_broadcast_engines(
            target_engine_indices=set(), mode="auto", **self.P1
        )
        self.assertEqual(out, frozenset())


class TestC9MixedStartupAssertion(unittest.TestCase):
    P1 = dict(train_gpu_ids=[0], infer_gpu_ids=[0, 1, 2], per_engine=1)

    def test_env_unset_is_noop(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(_ASSERT_MIXED_ENV, None)
            _assert_mixed_transport_startup(
                mode="cpu_serialize",
                pipeline_id="p",
                train_gpu_ids=None,
                infer_gpu_ids=None,
                per_engine=1,
            )  # no raise

    def test_mixed_overlap_topology_passes(self):
        with mock.patch.dict("os.environ", {_ASSERT_MIXED_ENV: "1"}):
            _assert_mixed_transport_startup(
                mode="auto", pipeline_id="p1", **self.P1
            )  # e0 colocate + e1/e2 broadcast → mixed, no raise

    def test_all_colocate_topology_fails_smoke_invalid(self):
        with mock.patch.dict("os.environ", {_ASSERT_MIXED_ENV: "1"}):
            with self.assertRaisesRegex(RuntimeError, "INVALID"):
                _assert_mixed_transport_startup(
                    mode="auto",
                    pipeline_id="p",
                    train_gpu_ids=[0, 1],
                    infer_gpu_ids=[0, 1],
                    per_engine=1,
                )

    def test_non_auto_mode_with_assert_env_fails(self):
        with mock.patch.dict("os.environ", {_ASSERT_MIXED_ENV: "1"}):
            with self.assertRaisesRegex(RuntimeError, "requires"):
                _assert_mixed_transport_startup(
                    mode="cpu_serialize", pipeline_id="p", **self.P1
                )


def _cfg(models):
    return types.SimpleNamespace(models=models)


def _model(name, num_gpus_per_engine=None, server_groups=None):
    return types.SimpleNamespace(
        name=name,
        num_gpus_per_engine=num_gpus_per_engine,
        server_groups=server_groups,
    )


def _group(num_gpus_per_engine=None):
    return types.SimpleNamespace(num_gpus_per_engine=num_gpus_per_engine)


class TestE10UniformityGuard(unittest.TestCase):
    def test_none_config_passes(self):
        _assert_uniform_engine_gpu_counts(None, 1)

    def test_matching_values_pass(self):
        cfg = _cfg([_model("actor", 2, [_group(2), _group(None)])])
        _assert_uniform_engine_gpu_counts(cfg, 2)

    def test_model_level_divergence_fails_naming_model(self):
        cfg = _cfg([_model("actor", 4)])
        with self.assertRaisesRegex(RuntimeError, "'actor'.*num_gpus_per_engine=4"):
            _assert_uniform_engine_gpu_counts(cfg, 2)

    def test_group_level_divergence_fails_naming_group(self):
        cfg = _cfg([_model("actor", 2, [_group(2), _group(4)])])
        with self.assertRaisesRegex(RuntimeError, r"server_groups\[1\].*num_gpus_per_engine=4"):
            _assert_uniform_engine_gpu_counts(cfg, 2)

    def test_unset_values_pass(self):
        cfg = _cfg([_model("actor", None, [_group(None)])])
        _assert_uniform_engine_gpu_counts(cfg, 3)


if __name__ == "__main__":
    unittest.main()
