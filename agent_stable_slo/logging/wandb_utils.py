import contextlib
import os
from typing import Any, Dict, Iterable, List, Optional


def _active() -> bool:
    return bool(os.getenv("WANDB_PROJECT"))


def ensure_online(require_online: bool = True) -> None:
    mode = os.getenv("WANDB_MODE", "").strip().lower()
    # Temporarily allow offline mode for faster data collection
    if mode == "offline":
        return  # Skip online enforcement in offline mode
    if require_online:
        if mode and mode != "online":
            raise RuntimeError(
                f"WANDB_MODE must be 'online' for this run. Got {mode!r}"
            )
        os.environ["WANDB_MODE"] = "online"


@contextlib.contextmanager
def maybe_run(
    name: str, config: Optional[Dict[str, Any]] = None, require_online: bool = False
):
    if not _active():
        yield None
        return
    if require_online:
        ensure_online(require_online=True)
    import wandb

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=name,
        config=config or {},
        dir=os.getenv("WANDB_DIR"),
        mode=os.getenv("WANDB_MODE", "online"),
    )
    try:
        yield run
    finally:
        try:
            run.finish()
        except Exception:
            pass


def init_run(
    name: str,
    project: str,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    require_online: bool = True,
):
    ensure_online(require_online=require_online)
    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        tags=tags or [],
        name=name,
        config=config or {},
        mode=os.getenv("WANDB_MODE", "online"),
    )


def log(run, metrics: dict, step: Optional[int] = None):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass


def log_artifact(
    run,
    path: str,
    name: str,
    type_: str = "dataset",
    aliases: Optional[Iterable[str]] = None,
):
    if run is None:
        return
    # Skip artifact upload when WANDB_SKIP_ARTIFACTS is set for faster runs
    if os.getenv("WANDB_SKIP_ARTIFACTS", "").lower() in ("1", "true", "yes"):
        return
    try:
        import wandb

        if not os.path.exists(path):
            return
        art = wandb.Artifact(name=name, type=type_)
        art.add_file(path, name=os.path.basename(path))
        if aliases:
            run.log_artifact(art, aliases=list(aliases))
        else:
            run.log_artifact(art)
    except Exception:
        pass


def create_episode_table():
    import wandb

    return wandb.Table(
        columns=[
            "task_id",
            "task_instance_id",
            "replicate_id",
            "decode_mode",
            "prompt_hash",
            "latency_ms",
            "tokens_in",
            "tokens_out",
            "retry_count",
            "repair_count",
            "candidate_count",
            "json_valid",
            "schema_valid",
            "clinc_intent_accuracy",
            "hotpot_answer_exact_match",
            "hotpot_answer_f1",
            "hotpot_faithfulness",
            "hotpot_contradiction_rate",
            "tool_success_rate",
            "success_at_slo",
            "raw_output",
        ]
    )


def add_episode_row(table, episode: Dict[str, Any]) -> None:
    if table is None:
        return
    metrics = episode.get("metrics", {})
    raw = episode.get("raw_output", "")
    table.add_data(
        episode.get("task_id"),
        episode.get("task_instance_id"),
        episode.get("replicate_id"),
        episode.get("decode_mode"),
        episode.get("prompt_hash"),
        episode.get("latency_ms"),
        episode.get("tokens_in"),
        episode.get("tokens_out"),
        episode.get("retry_count"),
        episode.get("repair_count"),
        episode.get("candidate_count"),
        metrics.get("json_valid"),
        metrics.get("schema_valid"),
        metrics.get("clinc_intent_accuracy"),
        metrics.get("hotpot_answer_exact_match"),
        metrics.get("hotpot_answer_f1"),
        metrics.get("hotpot_faithfulness"),
        metrics.get("hotpot_contradiction_rate"),
        metrics.get("tool_success_rate"),
        metrics.get("success_at_slo"),
        raw[:1000] if isinstance(raw, str) else raw,
    )


def log_table(run, table, key: str) -> None:
    if run is None or table is None:
        return
    try:
        run.log({key: table})
    except Exception:
        pass


def finish_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
