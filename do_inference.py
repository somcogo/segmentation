import os

import torch

from inference import do_inference_save_results
from models.mednext import MedNeXt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.set_num_threads(16)
comment = 'segmentation_2023-11-23_16.22.59_mednext-s-e3000-T3000-b4-lr0.001-XE1-adamw-augnnunet-imgsize-128-foregroundchance-0.5-gradacc-2.state'
save_path = os.path.join('saved_metrics', 'mednexthparams', 'inference', comment + 'overlap=0.25')
model_path = os.path.join('saved_models', 'mednexthparams', comment)
do_inference_save_results(save_path=save_path, image_size=(128, 128, 128), model_type='mednext-s', model_path=model_path, log=True, gaussian_weights=True, overlap=0.25)
