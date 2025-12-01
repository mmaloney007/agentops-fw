import os, platform, json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


def _import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _ram_gb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 1)
    except Exception:
        return None


def _cuda_props(torch_mod) -> Dict[str, Any]:
    props = {}
    try:
        props["name"] = torch_mod.cuda.get_device_name(0)
    except Exception:
        props["name"] = "cuda"
    try:
        props["capability"] = ".".join([str(x) for x in torch_mod.cuda.get_device_capability(0)])
    except Exception:
        props["capability"] = None
    try:
        props["total_mem_gb"] = round(
            torch_mod.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
        )
    except Exception:
        props["total_mem_gb"] = None
    try:
        props["supports_bf16"] = bool(torch_mod.cuda.is_bf16_supported())
    except Exception:
        props["supports_bf16"] = False
    return props


def _mps_props(torch_mod) -> Dict[str, Any]:
    props = {"name": "apple_mps", "capability": None, "supports_bf16": True}
    props["total_mem_gb"] = _ram_gb()
    try:
        # High watermark ratio env helps avoid MPS OOM thrashing
        props["mps_high_watermark_ratio"] = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "unset")
    except Exception:
        props["mps_high_watermark_ratio"] = "unset"
    return props


@dataclass
class HardwareInfo:
    kind: str
    device: str
    backend: str
    name: str
    total_mem_gb: Optional[float]
    capability: Optional[str]
    supports_bf16: bool
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [self.kind, self.name]
        if self.total_mem_gb:
            parts.append(f"{self.total_mem_gb}GB")
        if self.backend:
            parts.append(f"backend={self.backend}")
        parts.append(f"device={self.device}")
        if self.capability:
            parts.append(f"cap={self.capability}")
        if self.supports_bf16:
            parts.append("bf16")
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


def detect_hardware() -> HardwareInfo:
    torch_mod = _import_torch()
    if torch_mod and torch_mod.cuda.is_available():
        props = _cuda_props(torch_mod)
        name_lower = props.get("name", "").lower()
        kind = "nvidia_4090" if "4090" in name_lower else "nvidia_cuda"
        notes = "prefer bf16; use bitsandbytes if memory bound"
        return HardwareInfo(
            kind=kind,
            device="cuda:0",
            backend="cuda",
            name=props.get("name", "cuda"),
            total_mem_gb=props.get("total_mem_gb"),
            capability=props.get("capability"),
            supports_bf16=bool(props.get("supports_bf16")),
            notes=notes,
        )
    if torch_mod and getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        props = _mps_props(torch_mod)
        kind = "apple_mps"
        notes = "use 4/8-bit + Metal; keep batch small"
        return HardwareInfo(
            kind=kind,
            device="mps",
            backend="metal",
            name=props.get("name", "apple_mps"),
            total_mem_gb=props.get("total_mem_gb"),
            capability=props.get("capability"),
            supports_bf16=bool(props.get("supports_bf16", True)),
            notes=notes,
        )
    return HardwareInfo(
        kind="cpu_only",
        device="cpu",
        backend="cpu",
        name=platform.processor() or "cpu",
        total_mem_gb=_ram_gb(),
        capability=None,
        supports_bf16=False,
        notes="Torch CUDA/MPS unavailable",
    )


def recommended_defaults(info: Optional[HardwareInfo] = None) -> Dict[str, Any]:
    info = info or detect_hardware()
    cfg: Dict[str, Any] = {"device": info.device, "backend": info.backend}
    if info.backend == "cuda":
        cfg["torch_dtype"] = "bfloat16" if info.supports_bf16 else "float16"
        cfg["load_in_4bit"] = False
    elif info.backend == "metal":
        cfg["torch_dtype"] = "float16"
        cfg["load_in_4bit"] = True
    else:
        cfg["torch_dtype"] = "float32"
        cfg["load_in_4bit"] = True
    return cfg


if __name__ == "__main__":
    hw = detect_hardware()
    out = hw.as_dict()
    out["recommended"] = recommended_defaults(hw)
    print(json.dumps(out, indent=2))
