"""GRPO training loop with RLix scheduling hook integration points.

This is a structural stub that captures the 5-phase loop shape of the real
grpo_train() (which lives in the upstream NeMo RL repo and is ~3700 lines).
Its purpose is to:
  1. Make the DO_TIME_SHARING flag and hook call sites testable without the
     full NeMo RL dependency.
  2. Serve as the reference for where hooks must be inserted when the real
     grpo.py is imported as a submodule.

The five hook insertion points mirror Section 4.2 of nemo_rl_integration_plan.md.
"""
from __future__ import annotations

from typing import Any, Optional

from nemo_rl.algorithms.rlix_hooks import NoOpRLixHooks, RLixHooks

# Set to True when running under RLix multi-pipeline GPU time-sharing.
# When False, hooks default to NoOpRLixHooks and all scheduling calls are skipped,
# preserving identical behaviour to stock NeMo RL.
DO_TIME_SHARING: bool = False


def grpo_train(
    config: Any,
    *,
    hooks: Optional[RLixHooks] = None,
) -> None:
    """GRPO training loop with optional RLix scheduling hooks.

    Args:
        config: GRPOConfig (or any object with a num_steps attribute).
        hooks: RLixHooks implementation.  Defaults to NoOpRLixHooks so callers
            that do not use RLix need not pass anything.

    Hook call order per step:
        0. begin_progress_batch — record step's trajectory target (MUST precede before_generation
                                  so step_target_estimate is available when requesting GPUs)
        1. before_generation    — request inference GPU allocation
        2. [generation phase]   — prepare_for_generation → generate → finish_generation
           end_progress_batch   — called each mini-batch inside generation loop
        3. after_generation     — release inference GPU
        4. [advantage computation]
        5. before_training      — request training GPU allocation
        6. [training phase]     — policy.train(data)
        7. after_training       — release training GPU
        8. before_weight_sync   — expand sleeping inference workers
        9. [weight sync]        — refit_policy_generation(policy, policy_generation)
    """
    if hooks is None:
        hooks = NoOpRLixHooks()

    num_steps: int = getattr(config, "num_steps", 1)
    # In the real grpo_train this comes from config (train_batch_size, n_epochs, etc.)
    step_trajectory_target: int = getattr(config, "step_trajectory_target", 1)

    for step in range(num_steps):
        # ===== Task 6: record target before requesting GPUs =====
        # Must come before before_generation so _count_intended_for_step is set.
        # Task 7 will read hooks._count_intended_for_step as step_target_estimate
        # when calling scheduler.request_gpus().
        hooks.begin_progress_batch(step, step_trajectory_target)

        # ===== HOOK 1: request generation GPU =====
        hooks.before_generation(step)

        # [generation phase]
        # prepare_for_generation()
        # responses = generate(batch)
        # rewards = env.step(responses)
        # finish_generation()

        # ===== HOOK 2: release generation GPU =====
        hooks.after_generation(step)

        # [advantage computation]
        # advantages = compute_advantages(rewards)

        # ===== HOOK 3: request training GPU =====
        hooks.before_training(step)

        # [training phase]
        # policy.train(data)

        # ===== HOOK 4: release training GPU =====
        hooks.after_training(step)

        # ===== HOOK 5: prepare weight sync =====
        hooks.before_weight_sync(step)

        # [weight sync]
        # refit_policy_generation(policy, policy_generation)
