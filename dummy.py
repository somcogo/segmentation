from models.unetblocks import UNetConvBlock, UNetUpBlock
from models.swinunetr import SwinUNETR
import torch

unet = UNetConvBlock(in_size=1, out_size=2)
upblock = UNetUpBlock(in_size=16, out_size=8)
unetr = SwinUNETR(img_size=64,
		patch_size=8,
		in_chans=1,
		num_classes=2,
		embed_dim=48,
		window_size=2,
		depths=[2, 2],
		num_heads=[3, 6])
x = torch.rand(8, 1, 64, 64, 64)
y = torch.rand(8, 8, 32, 32, 32)
z = unetr(x)
print(z.size())
