import numpy as np
import torch
from scipy.ndimage import rotate

def crop_section_and_aug(image, mask, section, image_size, pad=30):
    rng = np.random.default_rng()
    H, W, D = image.shape
    h, w, d = image_size
    x_0 = rng.integers(H - h)
    y_0 = rng.integers(W - w)
    z_0 = rng.integers(D - d)
    x_1, y_1, z_1 = x_0 + h, y_0 + w, z_0 + d
    image = np.pad(image, pad, mode='edge')[x_0:x_1+2*pad, y_0:y_1+2*pad, z_0:z_1+2*pad]
    mask = np.pad(mask, pad, mode='constant', constant_values=0)[x_0:x_1+2*pad, y_0:y_1+2*pad, z_0:z_1+2*pad]

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