import json, os, platform, random, subprocess, sys, tempfile, time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be absent in some envs
    torch = None

try:
    import jax  # type: ignore
except Exception:  # pragma: no cover - jax is optional
    jax = None


def _git_rev() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _git_status() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "status", "--short", "--branch"], stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="ignore")
            .strip()
        )
    except Exception:
        return None


def _pip_freeze(max_lines: int = 200) -> List[str]:
    try:
        out = (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="ignore")
            .splitlines()
        )
        if len(out) > max_lines:
            return out[:max_lines] + [f"...truncated {len(out) - max_lines} lines"]
        return out
    except Exception:
        return []


def _cuda_env() -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"cuda_available": False}
    info: Dict[str, Any] = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
    }
    try:
        info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        info["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    except Exception:
        pass
    try:
        info["version"] = torch.version.cuda
    except Exception:
        pass
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return info


def env_snapshot(extra: Optional[Dict[str, Any]] = None, include_packages: bool = True) -> Dict[str, Any]:
    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_rev": _git_rev(),
        "git_status": _git_status(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": getattr(torch, "__version__", None),
        "jax": getattr(jax, "__version__", None) if jax else None,
        "cuda": _cuda_env(),
        "hf_cache": {
            "HF_HOME": os.getenv("HF_HOME"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            "HF_DATASETS_CACHE": os.getenv("HF_DATASETS_CACHE"),
        },
    }
    if include_packages:
        snap["pip_freeze"] = _pip_freeze()
    if extra:
        snap.update(extra)
    return snap


def set_seed(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
            try:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except Exception:
                pass
    if jax is not None:
        try:
            jax.random.PRNGKey(seed)
        except Exception:
            pass
    return capture_rng_state()


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        try:
            state["torch_cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    if torch is not None:
        try:
            if "torch_cpu" in state:
                torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
