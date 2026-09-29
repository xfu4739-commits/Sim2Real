import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_setup import cleanup, setup
from demo_checkpoint import demo_checkpoint
from ToyModel import demo_basic, run_demo


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        # 这里虽然实现了模型"并行"（拆分到多卡），但其实每次前向/反向还是串行计算：
        # 先在 dev0 跑 net1，再把数据搬去 dev1 跑 net2，并没有同一批数据同时在两张卡上算，只是显存分摊。
        # 如果想并行地同时让 dev0 和 dev1 都有工作负载（典型场景是流水线并行/pipeline parallel），
        # 需要引入流水线 minibatch scheduling，如 torch.distributed.pipeline.sync.Pipe，但这里只演示简单串联。
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        # x先在dev0上跑net1和relu
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        #再把结果搬到dev1上继续跑net2
        x = x.to(self.dev1)
        return self.net2(x)


def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    # 这一步是为每个进程分配两个不同的 GPU 设备，实现模型并行。
    # 假设总共有 n_gpus 个 GPU，每个进程负责两张卡（dev0, dev1）。
    # 比如 rank=0 时用 0,1 卡，rank=1 时用 2,3 卡，以此类推。
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()
