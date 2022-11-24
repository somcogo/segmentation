from torch.utils.data import DataLoader, Dataset
import torch

import copy
import glob
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import functools

def getIdList():
    raw_path_list = glob.glob('./data/altesCT/raw/*')
    id_list = [os.path.split(path)[-1][:-7] for path in raw_path_list]
    return id_list

def getCenterCoords(img):

    x_sum = np.sum(img, axis=(1, 2))
    x_indices = np.where(x_sum != 0)
    x_min = x_indices[0][0]
    x_max = x_indices[0][-1]

    y_sum = np.sum(img, axis=(0, 2))
    y_indices = np.where(y_sum != 0)
    y_min = y_indices[0][0]
    y_max = y_indices[0][-1]

    z_sum = np.sum(img, axis=(0, 1))
    z_indices = np.where(z_sum != 0)
    z_min = z_indices[0][0]
    z_max = z_indices[0][-1]

    x = int(( x_max + x_min) / 2)
    y = int(( y_max + y_min) / 2)
    z = int(( z_max + z_min) / 2)

    return x, y, z

@functools.lru_cache
def getImages(id):
    data = nib.load('./data/altesCT/raw/{}.nii.gz'.format(id))
    data_mask = nib.load('./data/altesCT/mask/{}-labels.nii'.format(id))
    img = data.get_fdata()
    mask = data_mask.get_fdata()

    zooms = data.header.get_zooms()
    center_coord = getCenterCoords(mask)
    lengths = ( np.asarray(50) / np.asarray(zooms) ).astype(int)
    img = img[
        center_coord[0] - lengths[0]: center_coord[0] + lengths[0],
        center_coord[1] - lengths[1]: center_coord[1] + lengths[1],
        center_coord[2] - lengths[2]: center_coord[2] + lengths[2]
    ]
    mask = mask[
        center_coord[0] - lengths[0]: center_coord[0] + lengths[0],
        center_coord[1] - lengths[1]: center_coord[1] + lengths[1],
        center_coord[2] - lengths[2]: center_coord[2] + lengths[2]
    ]

    interpolation_zoom = 64 / np.asarray(img.shape)
    img = ndimage.zoom(img, interpolation_zoom, order=1)
    mask = ndimage.zoom(mask, interpolation_zoom, order=1)
    return img, mask

class SegmentationDataset(Dataset):
    def __init__(self):
        self.id_list = copy.copy(getIdList())
    
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        id = self.id_list[index % 2]
        img, mask = getImages(id)
        img = torch.tensor(img)
        img = (img - torch.mean(img)) / torch.std(img)
        mask = torch.tensor(mask)
        mask = (mask - torch.mean(mask)) / torch.std(mask)
        return img, mask

def getDataLoader(batch_size):
    train_ds = SegmentationDataset()
    return DataLoader(train_ds, batch_size=batch_size, drop_last=True)