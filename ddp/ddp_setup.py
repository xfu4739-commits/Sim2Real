import os

import torch
import torch.distributed as dist


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.


def _distributed_backend():
    """Pick a process-group backend compatible with the installed PyTorch build."""
    # Windows DDP only supports gloo; do not touch torch.accelerator (2.x-only API).
    if os.name == "nt":
        return "gloo"
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _windows_file_init_method():
    init_file = os.environ.get("DDP_INIT_FILE")
    if not init_file:
        raise RuntimeError(
            "On Windows, export a shared DDP_INIT_FILE path before training "
            "(train.py sets this automatically before mp.spawn)."
        )
    normalized = os.path.abspath(init_file).replace("\\", "/")
    return f"file:///{normalized}"


def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    backend = _distributed_backend()
    if os.name == "nt":
        # Avoid TCPStore/libuv rendezvous issues on many Windows PyTorch builds.
        dist.init_process_group(
            backend,
            init_method=_windows_file_init_method(),
            rank=rank,
            world_size=world_size,
        )
        return

    os.environ.setdefault("TORCH_DISTRIBUTED_USE_LIBUV", "0")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
