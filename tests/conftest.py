import pytest


@pytest.fixture(autouse=True)
def _env_guard(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.test")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh")
    yield
