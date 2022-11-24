from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset

from collections import namedtuple
import glob
import csv
import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def getIdList():
    raw_path_list = glob.glob('./data/altesCT/raw/*')
    id_list = [os.path.split(path)[-1][:-7] for path in raw_path_list]
    return id_list

def getCenterList():
    raw_path_list = glob.glob('./data/verse19/dataset-verse19test/rawdata/*/*')
    raw_path_list.sort()
    raw_path_list = raw_path_list[:10]

    derivatives_path_list = glob.glob('./data/verse19/dataset-verse19test/derivatives/*')
    derivatives_path_list.sort()
    derivatives_path_list = derivatives_path_list[:10]

    id_list = [os.path.split(path)[-1][:-10] for path in raw_path_list]
    id_list.sort()
    id_list = id_list[:10]

    center_list = {}
    for id in id_list:
        with open('./data/verse19/dataset-verse19test/derivatives/{}/{}_seg-vb_ctd.json'.format(id, id)) as f:
            data = json.load(f)
            for x in data[1:]:
                if x['label'] == 20:
                    center_list[id] = [x['X'], x['Y'], x['Z']]
    
    return id_list, center_list

def getSegmentBoundary(img):

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

    return x_min, x_max, y_min, y_max, z_min, z_max

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
    print(img.shape)
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
    img = data.get_fdata()
    mask = data_mask.get_fdata()
    # mask = mask == 20
    boundary_coord = getSegmentBoundary(mask)

    img = interpolateImage(img, data.header.get_zooms())
    mask = interpolateImage(mask, data_mask.header.get_zooms())

    print(mask.shape)
    img = cropImage(img, boundary_coord)
    mask = cropImage(mask, boundary_coord)
    mask = mask.astype(float)

    print(mask.shape)
    # file = nib.Nifti1Image(img, np.eye(4))
    # file_mask = nib.Nifti1Image(mask, np.eye(4))
    # nib.save(file, os.path.join('./data/test', 'test.nii.gz'))
    # nib.save(file_mask, os.path.join('./data/test', 'test_mask.nii.gz'))
    # break

class SampleDataset(Dataset):
    def __init__(self, datalist):
        super().__init__()
        self.datalist = datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        return self.datalist[index]