from .swin_transformer_block import SwinTransformer
from .unetblocks import UNetConvBlock, UNetUpBlock
import torch.nn as nn

class SwinUNETR(nn.Module):
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

		self.swin_tr = SwinTransformer(
			img_size=img_size,
			patch_size=patch_size,
			in_chans=in_chans,
			num_classes=num_classes,
			embed_dim=embed_dim,
			window_size=window_size,
			depths=depths,
			num_heads=num_heads
		)
		self.encoders = nn.ModuleList()
		for i in range(len(depths) + 2):
			if i == 0:
				in_size = in_chans
				out_size = embed_dim
			else:
				in_size = embed_dim * 2 ** (i - 1)
				out_size = embed_dim * 2 ** (i - 1)
			encoder = UNetConvBlock(in_size=in_size, out_size=out_size)
			self.encoders.append(encoder)

		self.decoders = nn.ModuleList()
		for i in range(len(depths) + 1):
			if i == 0:
				in_size=embed_dim
				out_size=embed_dim
			else:
				in_size=embed_dim * 2 ** (i)
				out_size=embed_dim * 2 ** (i - 1)
			decoder = UNetUpBlock(in_size=in_size, out_size=out_size)
			self.decoders.append(decoder)

	def forward(self, x):
		hidden_features = self.swin_tr(x)
		enc = []
		dec = []
		for i in range(len(hidden_features) + 1):
			if i == 0:
				enc.append(self.encoders[0](x))
			else:
				enc.append(self.encoders[i](hidden_features[i - 1]))
		
		for i in range(len(hidden_features)):
			if i == 0:
				dec.append(self.decoders[i](enc[-1], hidden_features[-2]))
			else:
				dec.append(self.decoders[i](dec[i - 1], enc[-i - 1]))

		return dec[-1]