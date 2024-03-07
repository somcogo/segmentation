import numpy as np
import torch
from scipy.ndimage import rotate
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

def crop_section_and_aug(image, mask, image_size, pad=30, mode='trn', aug='nnunet', foreground_pref_chance=0., rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    pref_foreground = rng.uniform() < foreground_pref_chance
    H, W, D = image.shape
    h, w, d = image_size
    if pref_foreground and mask.sum() > 0:
        x_c, y_c, z_c = [arr.mean().astype(np.longlong) for arr in np.where(mask)]
        x_low, x_high = max(0, x_c - h), min(H - h, x_c)
        y_low, y_high = max(0, y_c - w), min(W - w, y_c)
        z_low, z_high = max(0, z_c - d), min(D - d, z_c)
    else:
        x_low, x_high = 0, H - h
        y_low, y_high = 0, W - w
        z_low, z_high = 0, D - d

    x_0 = rng.integers(x_low, x_high)
    y_0 = rng.integers(y_low, y_high)
    z_0 = rng.integers(z_low, z_high)
    x_1, y_1, z_1 = x_0 + h, y_0 + w, z_0 + d
    image = np.pad(image, pad, mode='edge')[x_0:x_1+2*pad, y_0:y_1+2*pad, z_0:z_1+2*pad]
    mask = np.pad(mask, pad, mode='constant', constant_values=0)[x_0:x_1+2*pad, y_0:y_1+2*pad, z_0:z_1+2*pad]

    if aug == 'nnunet':
        image_size = [size + 2 * pad for size in image_size]
        tr = get_training_transform(image_size) if mode == 'trn' else get_validation_transform()
        data_dict = tr(data=np.expand_dims(image, axis=(0, 1)), seg=np.expand_dims(mask, axis=(0, 1)))
        image = data_dict['data']
        mask = data_dict['seg']
    elif aug == 'old':
        image, mask = do_old_aug(rng, image, mask) if mode == 'trn ' else image, mask
        image = torch.from_numpy(np.expand_dims(image, (0, 1)).copy())
        mask = torch.from_numpy(np.expand_dims(mask, (0, 1)).copy())
    else:
        image = torch.from_numpy(np.expand_dims(image, (0, 1)).copy())
        mask = torch.from_numpy(np.expand_dims(mask, (0, 1)).copy())
 
    image = image.squeeze((0, 1))[pad:h+pad, pad:w+pad, pad:d+pad]
    mask = mask.squeeze((0, 1))[pad:h+pad, pad:w+pad, pad:d+pad]
    return image, mask

def do_old_aug(rng:np.random.Generator, image, mask):
    flip_axes = rng.uniform(size=3) < 0.5
    for i, i_flip in enumerate(flip_axes):
        if i_flip:
            image = np.flip(image, i)
            mask = np.flip(mask, i)

    do_rotation = rng.uniform() < 0.5
    if do_rotation:
        phi, theta = rng.random(size=2)*2*np.pi
        image = rotate(image, angle=theta, axes=(0,1), order=3, mode='nearest')
        mask = rotate(mask, angle=theta, axes=(0,1), order=0, mode='constant', cval=0)
        image = rotate(image, angle=phi, axes=(1,2), order=3, mode='nearest')
        mask = rotate(mask, angle=phi, axes=(1,2), order=0, mode='constant', cval=0)
    
    do_scaling = rng.uniform() < 0.5
    if do_scaling:
        scaling = rng.uniform(low=.7, high=1.3)
        image = scaling * image
    return image, mask

# TODO: cite nnUNet and batchgenerator https://github.com/MIC-DKFZ/nnUNet/tree/master https://github.com/MIC-DKFZ/batchgenerators
def get_training_transform(patch_size):
    tr_transforms = []
    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=False,
        do_rotation=True, angle_x=(-np.pi / 6, np.pi / 6), angle_y=(-np.pi / 6, np.pi / 6),
        angle_z=(-np.pi / 6, np.pi / 6), p_rot_per_axis=1,
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg='constant', border_cval_seg=-1, order_seg=1,
        random_crop=False,
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        p_independent_scale_per_axis=False))
    
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    tr_transforms.append(MirrorTransform((0, 1, 2)))
    
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_validation_transform():
    val_transforms = []

    val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms