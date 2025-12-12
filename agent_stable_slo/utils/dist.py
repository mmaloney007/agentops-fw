import os
import datetime
from typing import Tuple


def rank_world() -> Tuple[int, int]:
    try:
        rank = int(os.getenv("RANK", "0"))
    except Exception:
        rank = 0
    try:
        world = int(os.getenv("WORLD_SIZE", "1"))
    except Exception:
        world = 1
    if world < 1:
        world = 1
    if rank >= world:
        rank = 0
    return rank, world


def seed_with_rank(seed: int) -> int:
    rank, _ = rank_world()
    return seed + rank


def dist_available():
    try:
        import torch.distributed as dist  # type: ignore
        return dist.is_available()
    except Exception:
        return False


def dist_initialized():
    try:
        import torch.distributed as dist  # type: ignore
        return dist.is_initialized()
    except Exception:
        return False


def init_distributed(backend: str = "nccl", timeout_seconds: int = 1800) -> bool:
    if not dist_available():
        return False
    import torch.distributed as dist  # type: ignore
    if dist.is_initialized():
        return True
    rank, world = rank_world()
    if world <= 1:
        return False
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    return dist.is_initialized()


def barrier():
    if not dist_initialized():
        return
    try:
        import torch.distributed as dist  # type: ignore
        dist.barrier()
    except Exception:
        pass


def destroy_distributed():
    if not dist_initialized():
        return
    try:
        import torch.distributed as dist  # type: ignore
        dist.destroy_process_group()
    except Exception:
        pass
