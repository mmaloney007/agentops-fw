# Lessons Log

## Applied At Session Start

- Date: 2026-03-02
- Lesson: Run both repository-wide checks (for visibility) and changed-surface checks (for patch quality gate).
- Source: `/Users/maloney/.codex/tasks/lessons.md`

- Date: 2026-03-02
- Lesson: Prefer compatibility-safe codepaths and avoid assumptions about optional runtime dependencies.
- Source: `/Users/maloney/.codex/tasks/lessons.md`

## New Lessons

- Date: 2026-03-02
- Issue: Running Python wrapper scripts by file path failed to import top-level package modules.
- Root cause: `sys.path` started at `scripts/reasoning/` when launching the script directly, so repo root was not importable.
- Fix: Added repo-root bootstrap in `scripts/reasoning/run_reasoning_stack.py` before importing package modules.
- Prevention rule: For new script wrappers under `scripts/`, verify both invocation styles: `python script.py` and `python -m package.module`.
- Follow-up test added: End-to-end dry-run execution of `python scripts/reasoning/run_reasoning_stack.py --config configs/reasoning_stack/mlx_local.yaml`.
