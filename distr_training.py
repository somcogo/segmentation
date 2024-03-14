import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from main import SegmentationTrainingApp

def run(rank, world_size, config):
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)
    torch.cuda.device(rank)
    dist_arg = {'rank':rank,
                'world_size':world_size}
    trainer = SegmentationTrainingApp(distr=dist_arg, **config)
    trainer.main()

    dist.destroy_process_group()


def run_distr_training(config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run,
        args=(world_size, config),
        nprocs=world_size,
        join=True)