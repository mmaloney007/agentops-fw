import os, shutil, tempfile, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch  # type: ignore

from agent_stable_slo.utils.repro import capture_rng_state, restore_rng_state


class CheckpointManager:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.ckpt_root = self.out_dir / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def latest(self) -> Optional[Path]:
        ckpts = sorted(self.ckpt_root.glob("step_*"))
        return ckpts[-1] if ckpts else None

    def save(
        self,
        step: int,
        model,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        baseline: float,
        metadata: Dict[str, Any],
        scheduler: Optional[Any] = None,
    ) -> Path:
        ckpt_dir = self.ckpt_root / f"step_{step:06d}"
        tmp_dir = Path(tempfile.mkdtemp(prefix=ckpt_dir.name + "_", dir=self.ckpt_root))

        # Save adapter/tokenizer separately to keep state readable
        adapter_dir = tmp_dir / "adapter"
        tokenizer_dir = tmp_dir / "tokenizer"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(tokenizer_dir)

        state_path = tmp_dir / "state.pt"
        rng_state = capture_rng_state()
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "baseline": baseline,
                "rng_state": rng_state,
                "metadata": metadata,
                "saved_at": time.time(),
            },
            state_path,
        )

        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        os.replace(tmp_dir, ckpt_dir)
        return ckpt_dir

    def load(
        self,
        ckpt_dir: str,
        model,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        scheduler: Optional[Any] = None,
    ) -> Tuple[int, float, Dict[str, Any], Any]:
        ckpt_path = Path(ckpt_dir)
        state_path = ckpt_path / "state.pt"
        adapter_dir = ckpt_path / "adapter"
        tokenizer_dir = ckpt_path / "tokenizer"

        if not state_path.exists():
            raise FileNotFoundError(f"checkpoint missing state.pt at {ckpt_path}")
        if not adapter_dir.exists() or not tokenizer_dir.exists():
            raise FileNotFoundError(f"checkpoint missing adapter/tokenizer at {ckpt_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        # Load adapter/tokenizer
        adapter_bin = adapter_dir / "adapter_model.safetensors"
        if not adapter_bin.exists():
            adapter_bin = adapter_dir / "adapter_model.bin"
        if hasattr(model, "load_adapter"):
            try:
                model.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
                model.set_adapter("default")
            except Exception:
                model.load_state_dict(torch.load(adapter_bin, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load(adapter_bin, map_location="cpu"), strict=False)

        new_tok = tokenizer.__class__.from_pretrained(tokenizer_dir)
        tokenizer.__dict__.update(new_tok.__dict__)

        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass

        baseline = float(state.get("baseline", 0.0))
        restore_rng_state(state.get("rng_state", {}))
        return int(state.get("step", 0)), baseline, state.get("metadata", {}), tokenizer
