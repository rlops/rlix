# Rules for the Agent

- DO NOT add new files, like ".md" ".txt" or test, unless explicitly told or approved by user. 
- DO NOT make use of any new libraries or packages, unless explicitly told or approved by user.
- DO NOT display using table, use list or bulletin points instead. 
 
## 1. Safety (Critical)
- **No Secrets**: Never print, store, or commit tokens or keys.
- **No Destruction**: Do not delete files or force-push without asking first.
- **No Surprises**: Do not change files unrelated to the task.

## 2. Reliability
- **Happy Path First**: Assume perfect conditions. Do not fix edge case like hardware or external service failure unless asked (like network, hardware, server is down). Just document the risk and let the code fail.
- **Fail Fast**: Crash immediately with a clear error message; do not hide errors.
- **No Fault Tolerance**: Assume rare failures cannot happen to simplify implementation, ignore failure handling and fault tolerance cases by failing fast and loudly, then terminating the whole program. Do not implement retry logic, recovery mechanisms, or graceful degradation unless explicitly requested.
- **No Idempotent Operations**: Assume operations always succeed and execute exactly once. Do not design for idempotency, deduplication, or duplicate detection. Avoid idempotent operation patterns.
- **Check Your Work**: Run the smallest relevant test. If you can't, tell the user what command to run.

## 3. Scope & Quality
- **Small Changes**: Do the smallest change that solves the problem. No extra formatting.
- **Consistent Changes**: Keep the changes logically consistent across all files.
- **Reuse Code**: Don't reinvent the wheel; use existing patterns and libraries.

## 4. Communication
- **Simple English**: Use short sentences and plain words. Avoid jargon (or define it in one sentence).
- **Update Docs**: If you change behavior, paths, or commands, update docs in the same commit.
- **Ask First**: If unclear, ask one short question before starting.

## IDE Integration

IMPORTANT: Prefer use the `pycharm-index` MCP server when applicable for code navigation and refactoring:

IMPORTANT: When debugging, prefer using pycharm-debugger MCP tools to interact with the IDE debugger.


# Repository Guidelines

This repository is a multi-framework workspace for SchedRL design + integration across several RL/post-training stacks.

## Project Structure & Module Organization

- `design_doc/`: SchedRL design docs (protocols, adaptation plans).
- `third_party/`: third-party repos as git submodules.
- `third_party/nemo-rl/`, `third_party/nemo-gym/`: NeMo-RL and environment components.
- `third_party/ROLL/`: ROLL framework (Ray-based multi-role pipelines).
- `third_party/miles/`: Miles RL framework + rollout engines.
- `third_party/SkyRL/`: SkyRL train/agent framework.
- `third_party/sglang/`, `third_party/vllm/`: rollout engines.
- Each framework has its own packaging (`pyproject.toml` / `setup.py`) and its own `tests/` folder.

If a submodule folder is missing locally, run `git submodule update --init`.

## Build, Test, and Development Commands

Run commands from the relevant subproject root:

- ROLL: `cd third_party/ROLL && make test` (pytest) and `make precommit`.
- NeMo-RL (uses `uv`): `cd third_party/nemo-rl && uv sync` and `uv run --group test pytest -q`.
- Miles: `cd third_party/miles && pytest -q` (or follow `third_party/miles/docs/` and examples).
- SkyRL: `cd third_party/SkyRL &&` follow `third_party/SkyRL/README.md` and `third_party/SkyRL/skyrl-train/` examples.

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer explicit names over abbreviations.
- Follow the tooling and conventions of the subproject you’re changing:
  - `third_party/nemo-rl/`: `ruff` + `black` configured in `third_party/nemo-rl/pyproject.toml` (run via `uv`).
  - `third_party/ROLL/`: `pre-commit` hooks (`make precommit`).
- Keep edits scoped: avoid reformatting unrelated files.

## Testing Guidelines

- Use `pytest`; keep new tests next to the framework they cover under `*/tests/`.
- Prefer the smallest test that reproduces the behavior (unit → integration → e2e).

## Commit & Pull Request Guidelines

- Commit history here uses short, imperative summaries (e.g., `readme`); keep subjects concise.
- PRs should include: what changed, why, and which framework(s) it impacts.
- For protocol changes, update `design_doc/multi-pipeline-adaptation-plan.md` and keep `design_doc/archive/multi-pipeline_roll_old_design.md` as the reference sequence.
