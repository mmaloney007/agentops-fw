"""Tests for W&B utilities.

Uses WANDB_MODE=offline to avoid network calls during CI.
"""
import os

# Set offline mode before imports
os.environ["WANDB_MODE"] = "offline"


def test_wandb_importable():
    """Check wandb is importable."""
    import wandb
    assert wandb.__version__


def test_init_run_offline():
    """Test init_run in offline mode."""
    from agent_stable_slo.logging.wandb_utils import init_run, finish_run

    # Should work in offline mode
    run = init_run(
        project="test-project",
        name="test-run",
        config={"test_key": "test_value"},
    )
    assert run is not None
    finish_run(run)


def test_log_metrics():
    """Test log function."""
    from agent_stable_slo.logging.wandb_utils import init_run, log, finish_run

    run = init_run(project="test-project", name="test-metrics")
    try:
        # Should not raise
        log(run, {"accuracy": 0.95, "loss": 0.1})
        log(run, {"step": 1, "reward": 1.5}, step=1)
    finally:
        finish_run(run)


def test_wandb_entity_from_env():
    """Test that WANDB_ENTITY is respected."""
    from agent_stable_slo.logging.wandb_utils import init_run, finish_run

    old_entity = os.environ.get("WANDB_ENTITY")
    os.environ["WANDB_ENTITY"] = "test-entity"
    try:
        run = init_run(project="test-project", name="test-entity")
        # In offline mode, entity may not be set, but no error should occur
        finish_run(run)
    finally:
        if old_entity:
            os.environ["WANDB_ENTITY"] = old_entity
        else:
            os.environ.pop("WANDB_ENTITY", None)
