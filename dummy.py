from models.swin_transformer_block import SwinTransformerBlock, PatchEmbed, SwinTransformer
from models.unetblocks import UNetConvBlock, UNetUpBlock
import torch
from utils.data_loader import getDataLoader, getImages
import torch.nn as nn

trn, val = getDataLoader(
	batch_size=1,
	image_size=64,
	persistent_workers=True)

img, mask = getImages(4122560, 'altesCT', 64)
print(img.shape)
img, mask = getImages(4140811, 'altesCT', 64)
print(img.shape)
img, mask = getImages(4141097, 'altesCT', 64)
print(img.shape)
print('loaded successfully')
for i, batch in enumerate(trn):
    print(batch[0].shape)

