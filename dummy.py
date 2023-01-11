from models.swin_transformer_block import SwinTransformerBlock, PatchEmbed, SwinTransformer
from models.unetblocks import UNetConvBlock, UNetUpBlock
import torch
from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

pred = torch.Tensor([[[0,1],[0,1]],[[0,1],[0,1]]]).unsqueeze(dim=0)
m = torch.Tensor([[[1,1],[0,0]],[[1,1],[0,0]]]).unsqueeze(dim=0)
# n = 1-m
# m = m.unsqueeze(dim=0)
# n = n.unsqueeze(dim=0)

# c = torch.concat([m,n], dim=0)
# hd = HausdorffDistanceMetric(percentile=95)
hd_prediction = pred
hd_mask = m.unsqueeze(dim=0)
hd_neg_mask = 1 - hd_mask        
hd_ground_truth = torch.concat([hd_neg_mask, hd_mask], dim=0)
hd_dist = compute_hausdorff_distance(y_pred=pred, y=m, percentile=95)
print(hd_dist)