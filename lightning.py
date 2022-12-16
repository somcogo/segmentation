import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models.swin_transformer_block import SwinTransformer
from models.unetblocks import UNetConvBlock, UNetUpBlock
from models.swinunetr import SwinUNETR
from utils.data_loader import getDataLoader, getDataLoaderHDF5

import os
from time import time

t1 = time()

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
		self.swinunetr = SwinUNETR(img_size=img_size,
			patch_size=patch_size,
			in_chans=in_chans,
			num_classes=num_classes,
			embed_dim=embed_dim,
			window_size=window_size,
			depths=depths,
			num_heads=num_heads,
		)

		self.head = nn.Conv3d(in_channels=embed_dim, out_channels=2, kernel_size=1)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		features = self.swinunetr(x)
		out = self.head(features)
		loss_fn = nn.CrossEntropyLoss(reduction='sum')
		y = y.long()
		loss = loss_fn(out, y.squeeze(dim=1))
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		features = self.swinunetr(x)
		out = self.head(features)
		loss_fn = nn.CrossEntropyLoss(reduction='sum')
		y = y.long()
		loss = loss_fn(out.view(1, 2, -1), y.view(1, -1))
		self.log('val_loss', loss)

# data
image_size = 64
train_loader, val_loader = getDataLoaderHDF5(batch_size=1, image_size=image_size, num_workers=24)

# model
model = SwinUNETRModule(img_size=image_size, patch_size=2, embed_dim=12, depths=[2, 2], num_heads=[3, 6])

# logger
logger = TensorBoardLogger(save_dir=os.getcwd(), version=3, name="lightning_logs")

# training
trainer = pl.Trainer(
	accelerator='gpu',
	max_epochs=2,
	limit_train_batches=1.0,
	logger=logger,
	log_every_n_steps=10,
	)
trainer.fit(model, train_loader, val_loader)

t2 = time()
print(t2 - t1)
    
