from models.swin_transformer_block import SwinTransformerBlock, PatchEmbed, SwinTransformer
from models.unetblocks import UNetConvBlock, UNetUpBlock
import torch
from utils.data_loader import getIdList, getDataLoader
from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn

block = SwinTransformerBlock(
    dim=1,
    input_resolution=(8, 8, 8),
    num_heads=1,
    window_size=2,
    shift_size=1
)

embed = PatchEmbed(
    img_size=64,
    patch_size=8,
    in_chans=1
)

swin = SwinTransformer(
    img_size=256,
    patch_size=8,
    in_chans=1,
    num_classes=2,
    window_size=2,
)

swinunetr = SwinUNETR(
    img_size=256,
    in_channels=1,
    out_channels=1
)

unet = UNetConvBlock(in_size=1, out_size=16)
conv = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
up = nn.Upsample(scale_factor=2, mode='trilinear')
upblock = UNetUpBlock(in_size=1, out_size=16)
x = torch.rand(1, 1, 64, 64, 64)
y = torch.rand(1, 1, 128, 128, 128)
z = conv(x)
print(z[0].size())