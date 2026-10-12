import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_setup import cleanup, setup
from ToyModel import ToyModel


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # DDP 下各进程的模型参数始终一致（梯度会通过 all_reduce 同步），
        # 所以 只存一份就够了，避免多进程同时写同一文件。
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # barrier 是全员同步点：所有进程都到达这里才一起继续。
    # 确保 rank0 写完毕
    dist.barrier()

    # map_location 告诉 PyTorch：把原来在 cuda:0 上的张量，加载到当前 rank 对应的设备上。
    acc = torch.accelerator.current_accelerator()
    # 这里 acc 是 accelerator 的缩写，指当前所用的加速设备类型，例如 'cuda'、'cpu' 等。
    # configure map_location properly
    map_location = {f"{acc}:0": f"{acc}:{rank}"}
    
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")


if __name__ == "__main__":
    mp.spawn(demo_checkpoint, args=(2,), nprocs=2, join=True)
