# Avoid shadowing the stdlib `logging` module: import the real module from the
# stdlib path, mirror its symbols (including private ones), and register it in
# sys.modules so downstream libs (wandb, pytest, dill) see the expected API.
from importlib import util as _util
from sysconfig import get_paths as _get_paths
from pathlib import Path as _Path
import sys as _sys

_stdlib_path = _Path(_get_paths()["stdlib"]) / "logging" / "__init__.py"
try:
    _spec = _util.spec_from_file_location("stdlib_logging", _stdlib_path)
    _stdlib_logging = _util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_stdlib_logging)  # type: ignore[call-arg]
    globals().update(vars(_stdlib_logging))
    _sys.modules[__name__] = _stdlib_logging
except Exception:
    # Fallback: expose minimal placeholders to avoid AttributeError
    _LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for _lvl in _LEVELS:
        globals()[_lvl] = _lvl
