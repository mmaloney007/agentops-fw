# SpecSLOEval (Piece I) — reviewer-proof LaTeX bundle

This bundle contains:

- `main.tex` — revised paper draft (biblatex+biber)
- `refs.bib` — cleaned, *real* bibliography entries for all citations used in `main.tex`
- `tasks/` — the cited-summary JSONL suites referenced in the paper
- `tasks/schemas/summary_cite_schema.json` — JSON Schema used by the cited-summary tasks
- `figs/` — included artifacts (csv/png) used to support the pilot snapshot and figure generation
  - `figs/claimA1_mismatch.pdf` — main-body figure for Claim A1 (schema-validity vs Success@SLO mismatch)
  - `figs/latency_snapshot.csv` — the single-row artifact used to populate Table~\ref{tab:latency-snapshots}

## Build

```bash
make
```

This uses `latexmk` (runs `biber` automatically).

## Notes

- If you need to conform to a specific venue template (ACM/USENIX/IEEE), you may need to switch back to `natbib` and a `.bst` file. The paper is written to be portable, but the bibliography tooling is the most common point of template friction.


## Running against LM Studio vs vLLM

SpecSLOEval is designed to hit any OpenAI-compatible endpoint.

- **LM Studio**: start the OpenAI-compatible server from the UI (Developer → API Server).
- **vLLM (RTX 4090 recommended)**: run the OpenAI API server entrypoint, e.g.:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model <MODEL_NAME_OR_PATH> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Then set your client to `OPENAI_BASE_URL=http://localhost:8000/v1`.
