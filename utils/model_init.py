from monai.networks.nets import SwinUNETR as MonaiSwin

import torch
from models.swinunetr import SwinUNETR
from models.unet import UNet
from models.resnet50 import ResBottleneckBlock, ResNet
from models.mednext import MedNeXt

def model_init(model_type, swin_type=None, image_size=None, drop_rate=None, attn_drop_rate=None, ape=None, unet_depth=None, in_channels=1, pretrained=True):
    if model_type == 'swinunetr':
        if swin_type == 1:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=2, embed_dim=12, depths=[2, 2], num_heads=[3, 6], ape=ape)
        elif swin_type == 2:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=8, embed_dim=6, depths=[2, 2, 2], num_heads=[3, 6, 12], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == '2.2':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=8, embed_dim=12, depths=[2, 2, 2], num_heads=[3, 6, 12], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 3:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=4, embed_dim=3, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 4:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, window_size=4, patch_size=4, embed_dim=6, depths=[2, 2, 2], num_heads=[3, 6, 12], ape=ape)
        elif swin_type == 5:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=8, embed_dim=6, depths=[2, 2, 8], num_heads=[3, 6, 12], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd2s':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=8, window_size=8, embed_dim=12, depths=[2, 2], num_heads=[3, 6], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd2l':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=8, window_size=8, embed_dim=24, depths=[2, 2], num_heads=[3, 6], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd3s':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=4, window_size=4, embed_dim=6, depths=[2, 2, 2], num_heads=[3, 6, 12], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd3l':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=4, window_size=4, embed_dim=12, depths=[2, 2, 2], num_heads=[3, 6, 12], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd4s':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=2, window_size=8, embed_dim=3, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'd4l':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=2, window_size=8, embed_dim=6, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], ape=ape, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 6:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=9, embed_dim=6, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 7:
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=9, embed_dim=6, depths=[2, 2, 2], num_heads=[3, 6, 12], drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        elif swin_type == 'og':
            model = SwinUNETR(img_size=image_size, in_chans=in_channels, patch_size=2, window_size=8, embed_dim=48, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
    elif model_type == 'monaiswin':
        model = MonaiSwin(img_size=image_size, in_channels=in_channels, out_channels=2, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], feature_size=48, spatial_dims=3)
        if pretrained:
            swin_weights = torch.load('data/model_swinvit.pt')
            model.load_from(swin_weights)
    elif model_type == 'unet':
        model = UNet(in_channels=in_channels, n_classes=2, depth=unet_depth, wf=6, padding=True, batch_norm=True, up_mode='upsample')
    elif model_type == 'resnet':
        model = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], in_channels=in_channels, useBottleneck=True, outputs=2)
    elif model_type == 'mednext-s':
        model = MedNeXt(in_channels=1, n_channels=32, n_classes=2, exp_r=2, block_counts=[2,2,2,2,2,2,2,2,2], dim='3d')
    elif model_type == 'mednext-t1':
        model = MedNeXt(in_channels=1, n_channels=8, n_classes=2, exp_r=2, block_counts=[2,2,2,2,2,2,2,2,2], dim='3d')
    elif model_type == 'mednext-t2':
        model = MedNeXt(in_channels=1, n_channels=32, n_classes=2, exp_r=2, block_counts=[1,1,1,0,0,0,1,1,1], dim='3d')
    
    return model