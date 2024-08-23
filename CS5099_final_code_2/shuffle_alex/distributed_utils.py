import torch.distributed as dist
import torch
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Ensure the correct device is set for each process

def cleanup():
    dist.destroy_process_group()
