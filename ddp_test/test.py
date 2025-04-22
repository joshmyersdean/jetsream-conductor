import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class RandomDataset(Dataset):
    def __init__(self, size: int =1000, length: int=100) -> None:
        self.len = length
        self.data = torch.randn(length, size)
        self.targets = torch.randn(length, 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.len

def setup() -> None:
    dist.init_process_group("nccl")

def cleanup() -> None:
    dist.destroy_process_group()

def main() -> None:
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Simple model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    ).to(device)

    # Wrap model for distributed training
    model = DDP(model, device_ids=[local_rank])

    dataset = RandomDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):  # Short run
        sampler.set_epoch(epoch)
        for batch, targets in dataloader:
            batch, targets = batch.to(device), targets.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if dist.get_rank() == 0:
            print(f"Epoch {epoch} completed")

    cleanup()

if __name__ == "__main__":
    main()
