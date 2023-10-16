import numpy as np
import torch

def aug_tuple(image, mask):
    flip_axes = np.random.randint(low=0, high=2, size=3)
    for i, i_flip in enumerate(flip_axes):
        if i_flip > 0:
            image = np.flip(image, i).copy()
            mask = np.flip(mask, i).copy()
    rotate_number = np.random.randint(low=0, high=4, size=1)
    image = np.rot90(image, rotate_number, (0, 1))
    mask = np.rot90(mask, rotate_number, (0, 1))
    scaling = np.random.rand()
    scaling = (scaling * 0.2 ) + 0.9
    image = scaling * image
    return image, mask