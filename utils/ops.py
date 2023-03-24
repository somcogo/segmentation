import numpy as np
from scipy import ndimage

def aug_tuple(image, mask):
    flip_axes = np.random.randint(low=0, high=2, size=3)
    for i, i_flip in enumerate(flip_axes):
        if i_flip :
            image = np.flip(image, i)
            mask = np.flip(mask, i)
    rotate_degrees = np.random.randint(low=0, high=4, size=3)
    rotate_degrees = rotate_degrees * 90
    for i, i_degree in enumerate(rotate_degrees):
        image = ndimage.rotate(image, i_degree, axes=(i, (i + 1) % 3))
        mask = ndimage.rotate(mask, i_degree, axes=(i, (i + 1) % 3))
    scaling = np.random.rand()
    scaling = (scaling * 0.2 ) + 0.9
    image = scaling * image
    return image, mask