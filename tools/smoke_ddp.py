"""Minimal DDP runtime check, including gradient sync and device assignment.

Run from the repository root:
    torchrun --standalone --nproc-per-node=2 -m tools.smoke_ddp
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ddp.ddp_setup import cleanup
from train import init_distributed


def main():
    rank, world_size, device = init_distributed()
    try:
        model = nn.Linear(2, 1, bias=False).to(device)
        wrapped = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        sample = torch.tensor([[float(rank + 1), 1.0]], device=device)
        loss = wrapped(sample).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Synchronized gradients must leave every rank with identical weights.
        weight = model.weight.detach()
        low, high = weight.clone(), weight.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if not torch.allclose(low, high, atol=1e-6):
            raise RuntimeError('DDP parameters diverged between ranks')
        if rank == 0:
            print(f'DDP smoke test passed: world_size={world_size}, device={device}')
    finally:
        cleanup()


if __name__ == '__main__':
    main()
