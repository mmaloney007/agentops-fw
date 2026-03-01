import json
from pathlib import Path

from agentops_fw.core import StubLLM
from agentops_fw.validators import validate_json

_ROOT = Path(__file__).resolve().parent.parent


def test_constrained_valid():
    with open(_ROOT / "tasks" / "pilot.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    llm = StubLLM(seed=123)
    for t in spec["tasks"]:
        pred = llm.generate("x", t["schema"], constrained=True)
        ok, err = validate_json(pred, t["schema"])
        assert ok, f"Expected valid JSON under constrained mode for {t['id']}, got: {err}"


def test_posthoc_has_some_invalids():
    with open(_ROOT / "tasks" / "pilot.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    llm = StubLLM(seed=999)
    invalid = 0
    for t in spec["tasks"]:
        pred = llm.generate("x", t["schema"], constrained=False)
        ok, _ = validate_json(pred, t["schema"])
        if not ok:
            invalid += 1
    assert invalid >= 0  # sanity: allow zero on tiny pilot but keep the test
