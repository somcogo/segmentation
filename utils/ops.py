import numpy as np
import torch

def aug_tuple(image, mask):
    flip_axes = np.random.randint(low=0, high=2, size=3)
    for i, i_flip in enumerate(flip_axes):
        if i_flip:
            image = np.flip(image, i).copy()
            mask = np.flip(mask, i).copy()
    rotate_numbers = np.random.randint(low=0, high=4, size=3)
    for i, rotate_number in enumerate(rotate_numbers):
        image = np.rot90(image, rotate_number, (i, (i + 1)%3))
    scaling = np.random.rand()
    scaling = (scaling * 0.2 ) + 0.9
    image = scaling * image
    return image, mask