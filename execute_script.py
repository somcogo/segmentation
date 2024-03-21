import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from main import SegmentationTrainingApp
from utils.logconf import logging
from config import get_config

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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description="Train a model with or without distributed training.")
    parser.add_argument("--dist", default=False, type=bool, help="Run distributed training")
    parser.add_argument("--config", default='medcent', type=str, help="Config to use")
    args = parser.parse_args()

    config = get_config(args.config)
    
    if args.dist:
        run_distr_training(config)
    else:
        trainer = SegmentationTrainingApp(**config)
        trainer.main()

# model_path = '/home/hansel/developer/segmentation/saved_models/mednexthparams/segmentation_2023-11-23_16.22.59_mednext-s-e3000-T3000-b4-lr0.001-XE1-adamw-augnnunet-imgsize-128-foregroundchance-0.5-gradacc-2.state'
# mednext = SegmentationTrainingApp(epochs=1500, logdir='mednexthparams', lr=1e-3, batch_size=4, comment='2023-11-23_16.22.59-cont-mednext-s-e1500-T1500-b4-lr0.001-XE1-adamw-augnnunet-imgsize-128-foregroundchance-0.5-gradacc-2', loss_fn='XE', XEweight=1, optimizer_type='adamw', weight_decay=1e-5, model_type='mednext-s', aug='nnunet', scheduler_type='warmupcosine', T_0=1500, section='random', image_size=128, betas=(0.5, 0.9), foreground_pref_chance=0.5, grad_accumulation=2, model_path=model_path)
# mednext.main()

# unet = SegmentationTrainingApp(epochs=10, logdir='test', lr=1e-3, batch_size=2, comment='refactorimgs', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, model_type='unet', drop_rate=0.1, aug=True, scheduler_type='cosinewarmre', T_0=5000, unet_depth=3, section='random', image_size=144)
# unet.main()
# unet = SegmentationTrainingApp(epochs=1000, logdir='unethparams', lr=1e-3, batch_size=2, comment='unet-e1000-b2-lr0.001-XE16-adamw-aug-T_0-1000-section-random-imgsize-128-depth3', loss_fn='XE', XEweight=16, optimizer_type='adamw', weight_decay=1e-5, model_type='unet', aug=True, scheduler_type='cosinewarmre', T_0=1000, unet_depth=3, section='random')
# unet.main()

# epochs = 5000
# lr = 1e-3
# batch_size = 4
# grad_acc = 2
# foreground_chance = 0.5
# image_size = 128
# xe = 1
# swarm_training = False
# scheduler_type = 'warmupcosine'
# site_repetition = None
# iterations = None
# betas = (0.5, 0.9)
# swin = SegmentationTrainingApp(epochs=epochs, logdir='swinswarm', lr=lr, batch_size=batch_size, comment='swin-monaipretrained-e{}-T{}-b{}-lr{}-XE{}-adamw-augnnunet-warmup20cosine20-imgsize-{}-foregroundchance-{}-gradacc-{}-swarm{}-siterep-{}-iters-{}-beta-{}'.format(epochs, epochs, batch_size, lr, xe, image_size, foreground_chance, grad_acc, swarm_training, site_repetition, iterations, betas), loss_fn='XE', XEweight=xe, optimizer_type='adamw', weight_decay=1e-5, model_type='monaiswin', aug='nnunet', scheduler_type=scheduler_type, T_0=epochs, section='random', swin_type='og', pretrained=True, drop_rate=0.1, image_size=image_size, foreground_pref_chance=foreground_chance, grad_accumulation=grad_acc, swarm_training=swarm_training, site_repetition=site_repetition, iterations=iterations, betas=betas)
# swin.main()

# test = SegmentationTrainingApp(epochs=epochs, logdir='test', lr=lr, batch_size=batch_size, comment='swin-monaipretrained-e{}-T{}-b{}-lr{}-XE{}-adamw-augnnunet-warmup20cosine20-imgsize-{}-foregroundchance-{}-gradacc-{}'.format(epochs, epochs, batch_size, lr, xe, image_size, foreground_chance, grad_acc), loss_fn='XE', XEweight=xe, optimizer_type='adamw', weight_decay=1e-5, model_type='monaiswin', aug='nnunet', scheduler_type='warmupcosine', T_0=epochs, section='random', swin_type='og', pretrained=True, drop_rate=0.1, image_size=image_size, betas=(0.5, 0.9), foreground_pref_chance=foreground_chance, grad_accumulation=grad_acc, swarm_training=True)
# test.main()