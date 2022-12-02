import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models.swin_transformer_block import SwinTransformer
from models.unetblocks import UNetConvBlock, UNetUpBlock
from utils.data_loader import getDataLoader
from monai.networks.nets.swin_unetr import SwinUNETR

import os

class SwinUNETRModule(pl.LightningModule):
	def __init__(self,
		img_size=64,
		patch_size=8,
		in_chans=1,
		num_classes=2,
		embed_dim=48,
		window_size=2,
		depths=[2, 2, 2, 2],
		num_heads=[3, 6, 12, 24],
	):
		super().__init__()
		self.swin = SwinTransformer(
			img_size=img_size,
			patch_size=patch_size,
			in_chans=in_chans,
			num_classes=num_classes,
			embed_dim=embed_dim,
			window_size=window_size,
			depths=depths,
			num_heads=num_heads
		)
		# self.swinunetr = SwinUNETR(
		# 	img_size=64,
		# 	in_channels=1,
		# 	out_channels=1,
		# 	depths=(1, 1, 1, 1),
		# 	num_heads=(1, 1, 1, 1),
		# )
		self.encoders = nn.ModuleList()
		for i in range(depths + 2):
			if i == 0:
				in_size = in_chans
				out_size = embed_dim
			else:
				in_size = embed_dim * 2 ** (i - 1)
				out_size = embed_dim * 2 ** (i - 1)
			encoder = UNetConvBlock(in_size=in_size, out_size=out_size)
			self.encoders.append(encoder)

		self.decoders = nn.ModuleList()
		for i in range(depths + 1):
			if i == 0:
				in_size=embed_dim
				out_size=embed_dim
			else:
				in_size=embed_dim * 2 ** (i)
				out_size=embed_dim * 2 ** (i - 1)
			decoder = UNetUpBlock(in_size=in_size, out_size=out_size)
			self.decoders.append(decoder)

	def forward(self, x):
		x = self.swinunetr(x)

		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		hidden_features = self.swin(x)

		
		loss_fn = nn.MSELoss(reduction='mean')
		loss = loss_fn(x1, y)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x1, x2, x3 = self.swin(x)
		loss_fn = nn.MSELoss(reduction='mean')
		loss = loss_fn(x1, y)
		self.log('val_loss', loss)

# data
train_loader = getDataLoader(16)
val_loader = getDataLoader(16)

# model
model = SwinUNETRModule(img_size=64, patch_size=8, depths=[2, 2, 2], num_heads=[3, 3, 3])

# logger
logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

# training
trainer = pl.Trainer(
	accelerator='gpu',
	devices=1,
	max_epochs=5,
	limit_train_batches=0.5,
	logger=logger
	)
trainer.fit(model, train_loader, val_loader)
    
