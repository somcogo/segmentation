import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.norm1 = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.norm2(out)
        out += x
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv = nn.Conv3d(in_size, out_size, kernel_size=1)
        self.conv_block = UNetConvBlock(in_size, out_size)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.conv(up)
        out = torch.cat([up, bridge], 0)
        out = self.conv_block(out)

        return out