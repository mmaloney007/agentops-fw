
import os, contextlib
def _active(): return bool(os.getenv("WANDB_PROJECT"))
@contextlib.contextmanager
def maybe_run(name: str, config: dict | None = None):
    if not _active():
        yield None; return
    import wandb
    run = wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"),
                     name=name, config=config or {}, dir=os.getenv("WANDB_DIR"))
    try: yield run
    finally:
        try: run.finish()
        except Exception: pass
def log(run, metrics: dict, step: int | None = None):
    if run is None: return
    try: run.log(metrics, step=step)
    except Exception: pass
def log_artifact(run, path: str, name: str, type_: str = "dataset"):
    if run is None: return
    try:
        import wandb, os
        if not os.path.exists(path): return
        art = wandb.Artifact(name=name, type=type_); art.add_file(path, name=os.path.basename(path))
        run.log_artifact(art)
    except Exception: pass
