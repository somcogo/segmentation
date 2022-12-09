from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split

import copy
import glob
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import functools

def getTupleList():
    raw_path_list_alt = glob.glob('../data/altesCT_Segmentierung/*')
    raw_path_list_alt.sort()
    id_list_alt = [(os.path.split(path)[-1][:-14], 'altesCT') for path in raw_path_list_alt]
    raw_path_list_neu = glob.glob('../data/neuesCT_Segmentierung/*')
    raw_path_list_neu.sort()
    id_list_neu = [(os.path.split(path)[-1][:-14], 'neuesCT') for path in raw_path_list_neu]
    id_list = id_list_alt + id_list_neu
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
def getImages(id, scanner_version, image_size):
    data = nib.load('../data/{}/{}.nii.gz'.format(scanner_version, id))
    data_mask = nib.load('../data/{}_Segmentierung/{}-labels.nii.gz'.format(scanner_version, id))
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

    interpolation_zoom = image_size / np.asarray(img.shape)
    img = ndimage.zoom(img, interpolation_zoom, order=1)
    mask = ndimage.zoom(mask, interpolation_zoom, order=1)
    return img, mask

class SegmentationDataset(Dataset):
    def __init__(self, tuple_list, image_size):
        self.tuple_list = tuple_list
        self.image_size = image_size
    
    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, index):
        id = self.tuple_list[index][0]
        scanner_version = self.tuple_list[index][1]
        img, mask = getImages(id, scanner_version, image_size=self.image_size)
        img = torch.tensor(img)
        img = (img - torch.mean(img)) / torch.std(img)
        mask = torch.tensor(mask)
        # mask = (mask - torch.mean(mask)) / torch.std(mask)
        return img.unsqueeze(0).to(dtype=torch.float), mask.unsqueeze(0).to(dtype=torch.float)

def getDataLoader(batch_size, image_size):
    tuple_list = getTupleList()
    tuple_list = tuple_list[:20]
    train_list, val_list = train_test_split(tuple_list, test_size=0.1)
    train_ds = SegmentationDataset(train_list, image_size)
    val_ds = SegmentationDataset(val_list, image_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=48, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=48, drop_last=True)
    return train_dl, val_dl