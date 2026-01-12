from __future__ import annotations

import argparse
import sys

from agent_stable_slo.eval.p1_eval_harness import run_eval


def _build_eval_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("eval", help="Run P1 evaluation harness.")
    ap.add_argument("--criteria", required=True, help="Path to criteria YAML.")
    ap.add_argument("--suite", default="p1_core", help="Suite id from criteria.")
    ap.add_argument("--provider", default="lmstudio", help="Provider backend (lmstudio|ollama|vllm).")
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL (e.g., http://localhost:1234/v1).")
    ap.add_argument("--model", required=True, help="Policy model name/id.")
    ap.add_argument(
        "--mode",
        default="SPEC_DRIVEN",
        help="UNCONSTRAINED|PROVIDER_STRUCTURED|PROVIDER_STRUCTURED_PLUS_VALIDATE|SPEC_DRIVEN|SPEC_DRIVEN_PLUS_REPAIR|SPEC_DRIVEN_PLUS_SELFCONSISTENCY",
    )
    ap.add_argument("--max-examples", type=int, default=0, help="Optional max examples per task (0 = all).")
    ap.add_argument("--out-dir", default="out/p1_eval", help="Output directory for episodes/summary.")
    ap.add_argument("--stability-k", type=int, default=0, help="Override stability runs per prompt (0 = criteria default).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=2, help="Max retries for validation failures (P+V/S).")
    ap.add_argument("--repair-max-attempts", type=int, default=1, help="Max repair attempts for SPEC_DRIVEN_PLUS_REPAIR.")
    ap.add_argument("--self-consistency-samples", type=int, default=4)
    ap.add_argument("--self-consistency-max-ms", type=int, default=2500)
    ap.add_argument("--self-consistency-selection", default="majority_vote")
    ap.add_argument("--judge-base-url", default="", help="Override judge endpoint.")
    ap.add_argument("--judge-model", default="", help="Override judge model.")
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--disable-judge", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-tags", default="")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent Stable SLO CLI")
    sub = ap.add_subparsers(dest="command")
    _build_eval_parser(sub)

    args = ap.parse_args(argv)
    if args.command == "eval":
        run_eval(args)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
