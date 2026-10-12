import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_setup import cleanup, setup


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    '''
        一次性启动 world_size 个子进程，每个进程各自执行 demo_basic(rank, world_size)
    '''
    # mp.spawn 用于启动多个子进程并行执行指定的函数。下面参数含义说明：
    # demo_fn: 每个子进程要执行的函数，比如这里的 demo_basic
    # args: 传递给 demo_fn 的参数（不包括 rank，会自动作为第一个参数传入）
    # nprocs: 要启动的进程数，通常等于 world_size
    # join: 是否等待所有子进程执行完再返回，True 表示要等待
    mp.spawn(
        demo_fn,                  # 需要并行执行的函数，每个进程的 rank 会作为第一个参数传入
        args=(world_size,),       # demo_fn 剩余参数（除了 rank），这里传 world_size
        nprocs=world_size,        # 并行进程数，等于 world_size
        join=True,                # 等待所有子进程结束后再返回
    )


if __name__ == "__main__":
    run_demo(demo_basic, world_size=2)
