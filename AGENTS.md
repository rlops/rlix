# Rules for the Agent

- DO NOT add new files, like ".md" ".txt" or test, unless explicitly told or approved by user. 
- DO NOT make use of any new libraries or packages, unless explicitly told or approved by user.
- NEVER response with table, use list or Bullet points instead. 
- Human Understandability First: Code is rejected unless it is immediately maintainable by a teammate; every contribution MUST use intent-based naming, modular single-purpose functions with no magic values, and explicit type contracts that remove ambiguity.
- Typed Python Only: All new or modified Python code MUST annotate public APIs, minimize `Any` (use only for truly dynamic or untyped boundaries), and pass strict type checking (`mypy --strict`).
- ALWAYS comment code changes: MUST add comprehensive comments for short explanation first before changing any existing code.
- Copy-then-revise: ALWAYS try copy verbatim use `cp` or `sed` cmd then apply revision, noting the source file, when adding more than ~20 lines of new code. You MUST first search the codebase for a similar pattern (like use Grep/Glob for similar function or class).  Only write from scratch if no similar pattern exists.
- Trace root cause first: NEVER fix an error without identifying the root cause. If unclear, add targeted logs at suspicious spots and key points to observe runtime state before touching any code.     
- NO backaward compatiblity is needed when changes existing code.
- be general when adding new code, avoid adding code that is only for specific case or scenario, unless explicitly told or approved by user.
- avoid use getattr or hasattr `is not None`, unless explicitly told or approved by user.

## 1. Safety (Critical)
- **No Secrets**: Never print, store, or commit tokens or keys.
- **No Destruction**: Do not delete files or force-push without asking first.
- **No Surprises**: Do not change files unrelated to the task.

## 2. Reliability
- **Happy Path First**: Assume perfect conditions. Do not fix edge case like hardware or external service failure unless asked (like network, hardware, server is down). Just document the risk and let the code fail.
- **Fail Fast**: Crash immediately with a clear error message; do not hide errors.
- **No Fault Tolerance**: Assume rare failures cannot happen to simplify implementation, ignore failure handling and fault tolerance cases by failing fast and loudly, then terminating the whole program. Do not implement retry logic, recovery mechanisms, or graceful degradation unless explicitly requested or already exsit in the codebase.
- **No Idempotent Operations**: Assume operations always succeed and execute exactly once. Do not design for idempotency, deduplication, or duplicate detection. Avoid idempotent operation patterns.
- **Check Your Work**: Run the smallest relevant test. If you can't, tell the user what command to run.


## 3. Scope & Quality
- **Small Changes**: Do the smallest change that solves the problem. No extra formatting.
- **Consistent Changes**: Keep the changes logically consistent across all files.
- **Reuse Code**: Don't reinvent the wheel; use existing patterns and libraries.
- **No fallbacks**: NEVER creats fallbacks path unless explictely told or confirmed by user.

## 4. Communication
- **Always English**: Always think and respond to user in English.
- **Simple English**: Use short sentences and plain words in respnose. Avoid jargon (or define it in one sentence).
- **Update Docs**: If you change behavior, paths, or commands, update docs in the same commit.
- **Ask First**: If unclear, ask one short question before starting.

## 5. Code Understandability Rules
- **Name for Intent**: Use domain-specific names that explain purpose. NEVER use single-letter variable names — no exceptions, including loop indices.
- **Keep Functions Small**: Prefer single-responsibility functions; split functions that become hard to explain quickly.
- **No Magic Values**: Replace unexplained literals with well-named constants.
- **Readable Control Flow**: Prefer simple branches and early returns; avoid deep nesting and clever one-liners.
- **Enforce Strict Typing**: ALL functions/methods/classes MUST have explicit type annotations; use precise types, not vague ones. Use `Any` only at truly dynamic or untyped third-party boundaries.
- **Comments Explain Why**: Comprehensive comments/Google Style docstrings explain intent, reasons, constraints, and tradeoffs; replace magic values with named constants.
- **Follow Local Style Tools**: Keep formatting/linting consistent with the subproject so reviews focus on behavior, not style noise.

## 6. IDE Integration

IMPORTANT: Prefer use the `pycharm-index` MCP server when applicable for code navigation and refactoring:


# Repository Guidelines

This repository is a multi-framework workspace for SchedRL design + integration across several RL/post-training stacks.

default python env dir is /venv/main/bin/python


## Project Structure & Module Organization

- `design_doc/`: SchedRL design docs (protocols, adaptation plans).
- `external/`: third-party repos as git submodules.
- `external/nemo-rl/`, `external/nemo-gym/`: NeMo-RL and environment components.
- `external/ROLL_rlix/`: ROLL framework (Ray-based multi-role pipelines).
- `external/miles/`: Miles RL framework + rollout engines.
- `external/SkyRL/`: SkyRL train/agent framework.
- `external/sglang/`, `external/vllm/`: rollout engines.
- Each framework has its own packaging (`pyproject.toml` / `setup.py`) and its own `tests/` folder.

If a submodule folder is missing locally, run `git submodule update --init`.

## Build, Test, and Development Commands

Run commands from the relevant subproject root:

- ROLL: `cd external/ROLL_rlix && make test` (pytest) and `make precommit`.
- NeMo-RL (uses `uv`): `cd external/nemo-rl && uv sync` and `uv run --group test pytest -q`.
- Miles: `cd external/miles && pytest -q` (or follow `external/miles/docs/` and examples).
- SkyRL: `cd external/SkyRL &&` follow `external/SkyRL/README.md` and `external/SkyRL/skyrl-train/` examples.

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer explicit names over abbreviations.
- Follow the tooling and conventions of the subproject you’re changing:
  - `external/nemo-rl/`: `ruff` + `black` configured in `external/nemo-rl/pyproject.toml` (run via `uv`).
  - `external/ROLL_rlix/`: `pre-commit` hooks (`make precommit`).
- Keep edits scoped: avoid reformatting unrelated files.

## Testing Guidelines

- Use `pytest`; keep new tests next to the framework they cover under `*/tests/`.
- Prefer the smallest test that reproduces the behavior (unit → integration → e2e).

## Commit & Pull Request Guidelines

- Commit history here uses short, imperative summaries (e.g., `readme`); keep subjects concise.
- PRs should include: what changed, why, and which framework(s) it impacts.
- For protocol changes, update `design_doc/multi-pipeline-adaptation-plan.md` and keep `design_doc/archive/multi-pipeline_roll_old_design.md` as the reference sequence.
