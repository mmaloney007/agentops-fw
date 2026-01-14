# Plan

## Active
- [x] Housekeeping: archive legacy files, trim top-level, refresh docs and dependency notes.
- [ ] Phase 3: run full P1 evaluation suite (24 runs across 4 models x 6 modes).
- [ ] Phase 3: run P2 training experiments (baseline, naive RL, SLO-aware).
- [ ] Phase 4: update paper tables/figures and final polish.

## Considerations
- [ ] Archive `agentops_fw` if it is no longer part of the current workflow.
- [ ] Archive conda setup files (`environment.yml`, `activate_mamba.sh`) if mamba is no longer used.
- [ ] Archive Node files (`package.json`, `package-lock.json`) if no JS tooling remains.
- [ ] Decide whether to keep or archive community docs (`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`).

## Progress Log
- 2026-01-13: reset plan file to keep ongoing progress and keep top-level focused.
- 2026-01-13: archived legacy modules and older papers/results; added dependency notes; refreshed docs/CI paths.
- 2026-01-13: removed `docs/ONE_PAGER.md` and added follow-up cleanup considerations.
- 2026-01-13: archived top-level notebooks/data/figures, moved tooling into `scripts/`, and removed codex-named files from the root.
- 2026-01-13: enabled git-derived versioning via `setuptools-scm` in `pyproject.toml`.

## Notes
- Dependency sources: `pyproject.toml` is the package manifest; `requirements.txt` and `requirements-dev.txt` are for pip installs; `environment.yml` is for conda/mamba.
