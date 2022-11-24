from torchvision.transforms import ToTensor, Normalize, Compose
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

# @functools.lru_cache
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

def cropImage(img, boundary_coord):
    x1, x2, y1, y2, z1, z2 = boundary_coord
    x = int( (x2 + x1) / 2)
    y = int( (y2 + y1) / 2)
    z = int( (z2 + z1) / 2)
    max_size = max(x2 - x1, y2 - y1, z2 - z1)
    side_length = min(min(img.shape), max_size * 4)
    x_min = max( int(x - (side_length / 2)), 0)
    x_min = min( x_min, img.shape[0] - side_length)
    y_min = max( int(y - (side_length / 2)), 0)
    z_min = max( int(z - (side_length / 2)), 0)
    img = img[
        x_min: x_min + side_length,
        y_min: y_min + side_length,
        z_min: z_min + side_length
        ]
    return img

def getRatio(list):
    min = np.min(list)
    out = [l / min for l in list]
    return out

def interpolateImage(img, zooms):
    out = ndimage.zoom(img, zooms, order=1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # middle1 = int(img.shape[0] / 2)
    # middle2 = int(out.shape[0] / 2)
    # ax1.imshow(img[middle1,:,:])
    # ax2.imshow(out[middle2,:,:])
    # plt.show()

    return out

id_list = getIdList()
for id in id_list:
    data = nib.load('./data/altesCT/raw/{}.nii.gz'.format(id))
    data_mask = nib.load('./data/altesCT/mask/{}-labels.nii'.format(id))
    print('loaded images')
    img = data.get_fdata()
    mask = data_mask.get_fdata()
    print('got image data')
    img, mask = getImages(id)
    print(mask.shape, img.shape)
    # nifti_img = nib.Nifti1Image(img, np.eye(4))
    # nifti_mask = nib.Nifti1Image(mask, np.eye(4))
    # nib.save(nifti_img, os.path.join('/home/somahansel/work/segmentation_test/data/altesCT/test/{}.nii.gz'.format(id)))
    # nib.save(nifti_mask, os.path.join('/home/somahansel/work/segmentation_test/data/altesCT/test/{}-labels.nii.gz'.format(id)))

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