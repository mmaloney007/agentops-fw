#!/usr/bin/env python3
"""
Logging helpers for consistent driver + Dask worker/scheduler logs.
"""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_DASK_SHUTDOWN_NOISE = "Failed to communicate with scheduler during heartbeat"

_LOG_CONFIGURED = False
_FORWARD_HANDLER: logging.Handler | None = None
_WORKER_SHUTDOWN_IN_PROGRESS = False


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _library_log_level(levelno: int) -> int:
    if levelno <= logging.DEBUG:
        return logging.DEBUG
    return max(levelno, logging.WARNING)


def _set_library_loggers(
    app_levelno: int,
    dask_levelno: int | None = None,
    suppress_worker_info: bool = False,
) -> None:
    lib_level = (
        _library_log_level(app_levelno)
        if dask_levelno is None
        else _coerce_level(dask_levelno)
    )
    for name in (
        "dask",
        "distributed",
        "distributed.scheduler",
    ):
        logging.getLogger(name).setLevel(lib_level)

    # distributed.worker can be noisy with INFO messages when forwarding logs
    worker_level = max(lib_level, logging.WARNING) if suppress_worker_info else lib_level
    logging.getLogger("distributed.worker").setLevel(worker_level)

    # Suppress noisy shutdown/teardown messages from nanny and batched comms
    # These log ERROR-level tracebacks during normal worker lifecycle events
    for noisy_logger in ("distributed.nanny", "distributed.batched"):
        logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)


def _handler_level(app_levelno: int, dask_levelno: int | None = None) -> int:
    if dask_levelno is None:
        return app_levelno
    return min(app_levelno, _coerce_level(dask_levelno))


class _DaskWorkerShutdownFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        try:
            from distributed import get_worker  # type: ignore[import-not-found]
        except Exception:
            self._get_worker = None
        else:
            self._get_worker = get_worker

    def filter(self, record: logging.LogRecord) -> bool:
        if _DASK_SHUTDOWN_NOISE not in record.getMessage():
            return True
        if _WORKER_SHUTDOWN_IN_PROGRESS:
            return False
        if self._get_worker is None:
            return True
        try:
            worker = self._get_worker()
        except Exception:
            return True
        status = getattr(worker, "status", None)
        if status is None:
            return True
        status_text = str(status).lower()
        if "closing" in status_text or "closed" in status_text:
            return False
        return True


def _ensure_worker_shutdown_filter() -> None:
    worker_logger = logging.getLogger("distributed.worker")
    for existing in worker_logger.filters:
        if isinstance(existing, _DaskWorkerShutdownFilter):
            return
    worker_logger.addFilter(_DaskWorkerShutdownFilter())


def set_worker_shutdown_flag(enabled: bool) -> None:
    global _WORKER_SHUTDOWN_IN_PROGRESS
    _WORKER_SHUTDOWN_IN_PROGRESS = enabled


def setup_logging(
    level: str | int,
    *,
    dask_level: str | int | None = None,
    llm_level: str | int | None = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    force: bool = True,
) -> int:
    """
    Configure root logging once; safe to call multiple times.

    Args:
        level: Root logger level (app-wide default)
        dask_level: Separate level for dask/distributed loggers
        llm_level: Separate level for LLM operation loggers (neuralift_c360_prep.llm)
        fmt: Log format string
        datefmt: Date format string
        force: Force reconfiguration
    """
    global _LOG_CONFIGURED
    levelno = _coerce_level(level)
    dask_levelno = _coerce_level(dask_level) if dask_level is not None else None
    llm_levelno = _coerce_level(llm_level) if llm_level is not None else levelno
    handler_level = _handler_level(levelno, dask_levelno)
    if not _LOG_CONFIGURED:
        logging.basicConfig(level=levelno, format=fmt, datefmt=datefmt, force=force)
        _LOG_CONFIGURED = True
    else:
        root = logging.getLogger()
        root.setLevel(levelno)

    root = logging.getLogger()
    for handler in root.handlers:
        handler.setLevel(handler_level)

    _set_library_loggers(levelno, dask_levelno)

    # Configure LLM logger separately
    llm_logger = logging.getLogger("neuralift_c360_prep.llm")
    llm_logger.setLevel(llm_levelno)

    logging.captureWarnings(True)
    return levelno


def configure_remote_logging(
    level: str | int,
    dask_level: str | int | None = None,
    llm_level: str | int | None = None,
) -> None:
    """
    Ensure worker/scheduler logging is configured without clobbering handlers.
    """
    levelno = _coerce_level(level)
    dask_levelno = _coerce_level(dask_level) if dask_level is not None else None
    llm_levelno = _coerce_level(llm_level) if llm_level is not None else levelno
    handler_level = _handler_level(levelno, dask_levelno)
    root = logging.getLogger()
    root.setLevel(levelno)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATEFMT)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(handler_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(handler_level)
            if handler.formatter is None:
                handler.setFormatter(formatter)
    # Suppress worker INFO noise from out-of-band functions
    _set_library_loggers(levelno, dask_levelno, suppress_worker_info=True)

    # Configure LLM logger separately
    llm_logger = logging.getLogger("neuralift_c360_prep.llm")
    llm_logger.setLevel(llm_levelno)

    _ensure_worker_shutdown_filter()


def _log_on_scheduler(logger_name: str, levelno: int, message: str) -> None:
    logging.getLogger(logger_name).log(levelno, message)


class _SchedulerForwardingHandler(logging.Handler):
    def __init__(self, client: Any, name_prefix: str | None) -> None:
        super().__init__()
        self._client = client
        self._name_prefix = name_prefix

    def emit(self, record: logging.LogRecord) -> None:
        if self._name_prefix and not record.name.startswith(self._name_prefix):
            return
        status = getattr(self._client, "status", None)
        if status not in (None, "running"):
            return
        try:
            message = record.getMessage()
            if getattr(self._client, "asynchronous", False):
                loop = getattr(self._client, "loop", None)
                if loop is None or loop.is_closed():
                    return
                loop.add_callback(
                    self._client._run_on_scheduler,  # type: ignore[attr-defined]
                    _log_on_scheduler,
                    record.name,
                    record.levelno,
                    message,
                )
            else:
                self._client.run_on_scheduler(
                    _log_on_scheduler, record.name, record.levelno, message
                )
        except Exception:
            return


def attach_scheduler_log_forwarder(
    client: Any, *, level: str | int, name_prefix: str | None = "neuralift_c360_prep"
) -> None:
    """
    Forward driver logs into scheduler logs so Coiled captures them.

    Note: Only logs at the specified level or higher are forwarded to reduce
    scheduler traffic. Consider using WARNING+ to minimize overhead.
    """
    global _FORWARD_HANDLER
    root = logging.getLogger()
    if _FORWARD_HANDLER is not None:
        root.removeHandler(_FORWARD_HANDLER)

    handler = _SchedulerForwardingHandler(client, name_prefix=name_prefix)
    handler.setLevel(_coerce_level(level))
    root.addHandler(handler)
    _FORWARD_HANDLER = handler


def configure_dask_logging(
    client: Any,
    *,
    level: str | int,
    dask_level: str | int | None = None,
    llm_level: str | int | None = None,
    name_prefix: str | None = "neuralift_c360_prep",
    forward_to_scheduler: bool = True,
    forward_level: str | int = logging.WARNING,
) -> None:
    """
    Configure scheduler/worker logging and forward driver logs to scheduler.

    Args:
        client: Dask client
        level: Root logger level
        dask_level: Separate level for dask/distributed loggers
        llm_level: Separate level for LLM operation loggers
        name_prefix: Only forward logs from loggers with this prefix
        forward_to_scheduler: Whether to forward driver logs to scheduler
        forward_level: Minimum level for forwarded logs (default: WARNING to reduce overhead)
    """
    levelno = _coerce_level(level)
    dask_levelno = _coerce_level(dask_level) if dask_level is not None else None
    llm_levelno = _coerce_level(llm_level) if llm_level is not None else None
    try:
        client.run(configure_remote_logging, levelno, dask_levelno, llm_levelno)
    except Exception:
        pass
    try:
        client.run_on_scheduler(
            configure_remote_logging, levelno, dask_levelno, llm_levelno
        )
    except Exception:
        pass
    if forward_to_scheduler:
        forward_levelno = _coerce_level(forward_level)
        attach_scheduler_log_forwarder(
            client, level=forward_levelno, name_prefix=name_prefix
        )


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATEFMT",
    "setup_logging",
    "configure_remote_logging",
    "configure_dask_logging",
    "attach_scheduler_log_forwarder",
    "set_worker_shutdown_flag",
]
