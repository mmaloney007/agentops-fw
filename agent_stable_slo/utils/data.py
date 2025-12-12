import hashlib, json, os, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import Dataset  # type: ignore

from agent_stable_slo.utils.config import DatasetFingerprint


def _file_sha256(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_tasks(tasks_path: str) -> DatasetFingerprint:
    schema_hashes: Dict[str, str] = {}
    rows: List[dict] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            schema_path = rec.get("schema_path")
            if schema_path and os.path.exists(schema_path) and schema_path not in schema_hashes:
                schema_hashes[schema_path] = _file_sha256(schema_path)
            rows.append(rec)
    return DatasetFingerprint(
        tasks_path=tasks_path,
        sha256=_file_sha256(tasks_path),
        num_records=len(rows),
        schema_sha256=schema_hashes,
    )


def validate_fingerprint(fp: DatasetFingerprint, expected_sha256: Optional[str], allow_drift: bool = False) -> bool:
    if expected_sha256 and fp.sha256 != expected_sha256:
        msg = f"dataset hash mismatch: expected {expected_sha256}, got {fp.sha256}"
        if allow_drift:
            return False
        raise ValueError(msg)
    return True


def cache_dataset(tasks_path: str, cache_dir: str) -> Tuple[str, DatasetFingerprint]:
    """
    Copy tasks + schema files into a content-addressed cache directory so runs
    are reproducible even if the source files change mid-run.
    """
    fp = fingerprint_tasks(tasks_path)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{Path(tasks_path).stem}_{fp.sha256[:8]}"
    target.mkdir(parents=True, exist_ok=True)

    # Copy tasks file
    cached_tasks = target / Path(tasks_path).name
    if not cached_tasks.exists():
        shutil.copy2(tasks_path, cached_tasks)

    # Copy schemas
    for schema_path in fp.schema_sha256.keys():
        dst = target / Path(schema_path).name
        if not dst.exists():
            shutil.copy2(schema_path, dst)

    # Rewrite tasks file to point at cached schemas for portability
    rewritten = []
    with open(cached_tasks, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            schema_name = Path(rec["schema_path"]).name
            rec["schema_path"] = str((target / schema_name).resolve())
            rewritten.append(rec)
    with open(cached_tasks, "w", encoding="utf-8") as f:
        for rec in rewritten:
            f.write(json.dumps(rec) + "\n")

    cached_fp = fingerprint_tasks(str(cached_tasks))
    return str(cached_tasks), cached_fp


def load_jsonl_dataset(path: str) -> Dataset:
    rows = [json.loads(x) for x in open(path, "r", encoding="utf-8") if x.strip()]
    return Dataset.from_list(rows)
