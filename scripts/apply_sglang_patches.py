#!/usr/bin/env python3
"""Apply runbook Step 8 SGLang compat patches to miles sglang_engine.py.

Environment-only patches (new SGLang session-based weight-update API +
flush_cache fault tolerance). Idempotent: skips already-applied hunks.
"""
import sys

PATH = "/root/miles/miles/backends/sglang_utils/sglang_engine.py"

HUNKS = [
    # Patch 1a: expanded io_struct imports
    (
        """        from sglang.srt.entrypoints.http_server import _global_state
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from sglang.srt.utils import MultiprocessingSerializer
""",
        """        from sglang.srt.entrypoints.http_server import _global_state
        from sglang.srt.managers.io_struct import (
            BeginWeightUpdateReqInput,
            EndWeightUpdateReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.srt.utils import MultiprocessingSerializer
""",
    ),
    # Patch 1b: flush_cache=False in UpdateWeightsFromTensorReqInput
    (
        """        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=True,
        )
""",
        """        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=False,
        )
""",
    ),
    # Patch 1c: begin/end weight-update session wrapping
    (
        """        try:
            success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
                obj, None
            )
        except Exception as exc:  # noqa: BLE001
""",
        """        try:
            await _global_state.tokenizer_manager.begin_weight_update(
                BeginWeightUpdateReqInput(), None
            )
            success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
                obj, None
            )
            await _global_state.tokenizer_manager.end_weight_update(
                EndWeightUpdateReqInput(), None
            )
        except Exception as exc:  # noqa: BLE001
""",
    ),
    # Patch 2a: 400 retry inside flush_cache loop
    (
        """                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
""",
        """                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
                if response.status_code == 400:
                    logger.info("flush_cache returned 400, retrying in 1s...")
                    time.sleep(1)
                    continue
""",
    ),
    # Patch 2b: timeout -> warning
    (
        """        else:
            raise TimeoutError("Timeout while flushing cache.")
""",
        """        else:
            logger.warning("flush_cache timed out after 60 attempts, proceeding anyway")
""",
    ),
]

src = open(PATH).read()
applied, skipped = 0, 0
for old, new in HUNKS:
    if new in src:
        skipped += 1
        continue
    if old not in src:
        print(f"FATAL: hunk not found and not applied:\n{old[:120]}...")
        sys.exit(1)
    if src.count(old) != 1:
        print(f"FATAL: hunk not unique ({src.count(old)} occurrences):\n{old[:120]}...")
        sys.exit(1)
    src = src.replace(old, new)
    applied += 1

open(PATH, "w").write(src)
print(f"PATCH_OK applied={applied} skipped={skipped}")
import py_compile
py_compile.compile(PATH, doraise=True)
print("COMPILE_OK")
