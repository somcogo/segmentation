import os
import glob
import time
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from main import SegmentationTrainingApp
from utils.logconf import logging
from distr_training import run_distr_training

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

epochs = 5
batch_size = 2
grad_acc = 2
foreground_chance = 0.5
image_size = 128
dataset = 'median'
comm = 'test'
logdir = 'test'

med_cent_config = {'epochs':epochs,
          'logdir':logdir,
          'lr':1e-3,
          'batch_size':batch_size,
          'loss_fn':'XE',
          'XEweight':1,
          'optimizer_type':'adamw',
          'weight_decay':1e-5,
          'model_type':'mednext-s',
          'aug':'nnunet',
          'grad_accumulation':grad_acc,
          'foreground_pref_chance':foreground_chance,
          'image_size':image_size,
          'swarm_training':False,
          'scheduler_type':'warmupcosine',
          'T_0':epochs,
          'section':'random',
          'site_repetition':None,
          'iterations':None,
          'betas':(0.5, 0.9),
          'dataset':dataset,
          'comment':f'{comm}-mednext-s-e{epochs}-T{epochs}-b{batch_size}-lr1e-3-XE1-adamw-augnnunet-imgsize-{image_size}-foregroundchance-{foreground_chance}-gradacc-{grad_acc}-swarm-True-siterep-None-iters-None-beta-(0.5,0.9)-dataset-{dataset}'.replace(' ', '')}

def get_config(arg):
    match arg:
        case 'medcent':
            config = med_cent_config

    return config

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    torch.set_num_threads(2)
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