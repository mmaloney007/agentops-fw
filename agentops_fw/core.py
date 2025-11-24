import json, os, csv, random, time
from typing import Dict, Any, List, Tuple, Callable, Optional
from .validators import validate_json

# --- Simple plugin registry ---
_PLUGINS = {}
def plugin(name: str):
    def wrap(fn: Callable):
        _PLUGINS[name] = fn
        return fn
    return wrap

def get_plugin(name: str) -> Callable:
    if name not in _PLUGINS:
        raise ValueError(f"Unknown mode '{name}'. Available: {list(_PLUGINS)}")
    return _PLUGINS[name]

# --- Stub LLM (replace with your vLLM call) ---
class StubLLM:
    def __init__(self, seed: int = 13):
        self.rng = random.Random(seed)

    def generate(self, prompt: str, schema: Dict[str, Any], constrained: bool = False) -> Dict[str, Any]:
        props = list(schema.get("properties", {}).keys())
        # two special-case schemas used in pilot
        if set(props) == {"title", "bullets"}:
            out = {"title": "AlphaWidget Pro Launch", "bullets": ["Improved battery life", "Dust resistance"]}
        elif set(props) == {"name", "email"}:
            out = {"name": "Jane Q. Doe", "email": "jane.q.doe@example.org"}
        else:
            out = {}
            for k, v in schema.get("properties", {}).items():
                t = v.get("type")
                if t == "string": out[k] = "example"
                elif t == "integer": out[k] = 1
                elif t == "number": out[k] = 1.0
                elif t == "array": out[k] = ["item1", "item2"]
                elif t == "object": out[k] = {"key": "value"}
                else: out[k] = None

        # inject a validation error occasionally in post-hoc mode
        if not constrained:
            self.rng.seed(len(prompt) + len(str(schema)))
            if self.rng.random() < 0.3:
                if "email" in out:
                    out["email"] = "invalid_at_example.org"
                else:
                    out["extra"] = "oops"
        return out

# --- W&B helper ---
def maybe_log_wandb(rows: List[Dict[str, Any]], project: str, table_name: str):
    try:
        import wandb, pandas as pd
        wandb.init(project=project, reinit=True)
        df = pd.DataFrame(rows)
        wandb.log({table_name: wandb.Table(dataframe=df)})
        wandb.finish()
    except Exception as e:
        print(f"[info] W&B not available or failed to log: {e}")

# --- Modes ---
@plugin("posthoc")
def run_posthoc(llm: StubLLM, prompt: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str, float]:
    t0 = time.time()
    pred = llm.generate(prompt, schema, constrained=False)
    ok, err = validate_json(pred, schema)
    latency_ms = (time.time() - t0) * 1000.0
    return pred, ok, err, latency_ms

@plugin("constrained")
def run_constrained(llm: StubLLM, prompt: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str, float]:
    t0 = time.time()
    pred = llm.generate(prompt, schema, constrained=True)
    ok, err = validate_json(pred, schema)
    latency_ms = (time.time() - t0) * 1000.0
    return pred, ok, err, latency_ms

@plugin("contracts")
def run_contracts(llm: StubLLM, prompt: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str, float]:
    # Toy policy: use constrained to emulate pre/post enforcement
    t0 = time.time()
    pred = llm.generate(prompt, schema, constrained=True)
    ok, err = validate_json(pred, schema)
    if ok and isinstance(pred, dict):
        pred.setdefault("_policy", "contracts_v1")
    latency_ms = (time.time() - t0) * 1000.0
    return pred, ok, err, latency_ms

@plugin("budgeted")
def run_budgeted(llm: StubLLM, prompt: str, schema: Dict[str, Any], budget: int = 3, early_exit: bool = True) -> Tuple[Dict[str, Any], bool, str, float]:
    # Sample budgeted self-consistency (toy): try a few post-hoc samples; repair with constrained
    for i in range(budget):
        pred = llm.generate(prompt, schema, constrained=False)
        ok, _ = validate_json(pred, schema)
        if early_exit and ok: break
    # final "repair"
    t0 = time.time()
    final = llm.generate(prompt, schema, constrained=True)
    ok, err = validate_json(final, schema)
    latency_ms = (time.time() - t0) * 1000.0
    return final, ok, err, latency_ms

def run_tasks(taskfile: str, mode: str, out: str, project: str = "agentops-fw"):
    with open(taskfile, "r", encoding="utf-8") as f:
        spec = json.load(f)
    llm = StubLLM(seed=13)
    if mode not in _PLUGINS:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(_PLUGINS)}")

    rows = []
    for t in spec["tasks"]:
        prompt = f"Task: {t['family']}\nINPUT:\n{t['input']}\nOUTPUT: JSON"
        schema = t["schema"]
        pred, ok, err, latency_ms = _PLUGINS[mode](llm, prompt, schema)
        rows.append({
            "task_id": t["id"],
            "mode": mode,
            "valid": ok,
            "error": err,
            "latency_ms": round(latency_ms, 3),
            "prediction": json.dumps(pred, ensure_ascii=False)
        })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {out} with {len(rows)} rows")
    maybe_log_wandb(rows, project, f"{mode}_eval")
