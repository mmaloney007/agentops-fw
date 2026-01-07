import os

import pytest


@pytest.fixture(autouse=True)
def _env_guard(monkeypatch):
    if os.getenv("NL_INTEGRATION") == "1":
        yield
        return

    defaults = {
        "OPENAI_API_KEY": "test",
        "NL_SKIP_LLM": "1",
        "WANDB_API_KEY": "test",
        "DATABRICKS_HOST": "https://workspace.test",
        "DATABRICKS_CLIENT_ID": "client-id",
        "DATABRICKS_CLIENT_SECRET": "client-secret",
        "DATABRICKS_WAREHOUSE_ID": "wh",
        "DATABRICKS_OAUTH_ACCESS_TOKEN": "test",
    }
    for key, value in defaults.items():
        if os.getenv(key) is None:
            monkeypatch.setenv(key, value)
    yield
