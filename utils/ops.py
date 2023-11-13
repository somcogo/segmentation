import numpy as np
import torch
from scipy.ndimage import rotate

def crop_section_and_aug(image, mask, section, image_size, pad=30, aug=False, foreground_pref_chance=0.):
    rng = np.random.default_rng()
    pref_foreground = rng.uniform() < foreground_pref_chance
    H, W, D = image.shape
    h, w, d = image_size
    if pref_foreground:
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

    if aug:
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

    image = image[pad:h+pad, pad:w+pad, pad:d+pad]
    mask = mask[pad:h+pad, pad:w+pad, pad:d+pad]
    return image, mask