import copy
import glob
import os
import functools

import nibabel as nib
import h5py
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import torch
from sklearn.model_selection import train_test_split

from utils.ops import aug_tuple

class DatasetV2(Dataset):
    def __init__(self, data_path, mode, aug=False, section='large'):
        super().__init__()
        file = h5py.File(os.path.join(data_path, 'segmentation_{}.hdf5'.format(mode)), 'r')
        self.mode = mode
        self.section = section
        self.img_ds = file['img']
        self.mask_ds = file['mask']
        self.id_list = file.attrs['id_list']
        self.aug = aug

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        if self.section == 'large':
            x1, x2, y1, y2, z1, z2 = [6, 294, 6, 294, 0, 144]
        elif self.section == 'random':
            rng = np.random.default_rng()
            x1 = rng.integers(0, 156)
            y1 = rng.integers(0, 156)
            z1 = rng.integers(0, 6)
            x2 = x1 + 144
            y2 = y1 + 144
            z2 = z1 + 144
        image = np.array(self.img_ds[index, x1:x2, y1:y2, z1:z2])
        mask = np.array(self.mask_ds[index, x1:x2, y1:y2, z1:z2])
        img_id = self.id_list[index]
        if self.aug:
            image, mask = aug_tuple(image, mask)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0), img_id.tolist()
    
def getDataLoaderv2(batch_size, persistent_workers=True, aug=True, section='large'):
    trn_ds = DatasetV2(data_path='data', mode='trn', aug=aug, section=section)
    val_ds = DatasetV2(data_path='data', mode='val', aug=False, section=section)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    return trn_dl, val_dl


class NewDataset(Dataset):
    def __init__(self, data_path, mode, aug=False):
        super().__init__()
        alt_file = h5py.File(os.path.join(data_path, 'alt.hdf5'), 'r')
        neu_file = h5py.File(os.path.join(data_path, 'neu.hdf5'), 'r')
        if mode == 'trn':
            alt_start_ndx = 0
            alt_end_ndx = 138
            neu_start_ndx = 0
            neu_end_ndx = 114
        else:
            alt_start_ndx = 138
            alt_end_ndx = 172
            neu_start_ndx = 114
            neu_end_ndx = 143
        self.alt_img_ds = np.array(alt_file['img'])[alt_start_ndx:alt_end_ndx]
        self.neu_img_ds = np.array(neu_file['img'])[neu_start_ndx:neu_end_ndx]
        self.alt_mask_ds = np.array(alt_file['mask'])[alt_start_ndx:alt_end_ndx]
        self.neu_mask_ds = np.array(neu_file['mask'])[neu_start_ndx:neu_end_ndx]
        self.data = np.concatenate([self.alt_img_ds, self.neu_img_ds], axis=0)
        self.mask = np.concatenate([self.alt_mask_ds, self.neu_mask_ds], axis=0)
        alt_id_ds = np.array(alt_file['patient_id'][alt_start_ndx:alt_end_ndx], dtype=int)
        neu_id_ds = np.array(neu_file['patient_id'][neu_start_ndx:neu_end_ndx], dtype=int)
        self.id_list = np.concatenate([alt_id_ds, neu_id_ds], axis=0, dtype=int)
        self.aug = aug

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        image = self.data[index]
        mask = self.mask[index]
        img_id = self.id_list[index]
        if self.aug:
            image, mask = aug_tuple(image, mask)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0), img_id
    
def getNewDataLoader(batch_size, persistent_workers=True, aug=True):
    trn_ds = NewDataset(data_path='data', mode='trn', aug=aug)
    val_ds = NewDataset(data_path='data', mode='val', aug=False)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    return trn_dl, val_dl

def getTupleListHDF5():
    raw_path_list_alt = glob.glob('../../data/segmentation/alt/*')
    raw_path_list_alt.sort()
    id_list_alt = [(os.path.split(path)[-1][:-5], 'altesCT') for path in raw_path_list_alt]
    raw_path_list_neu = glob.glob('../../data/segmentation/neu/*')
    raw_path_list_neu.sort()
    id_list_neu = [(os.path.split(path)[-1][:-5], 'neuesCT') for path in raw_path_list_neu]
    id_list = id_list_alt + id_list_neu
    return id_list_alt, id_list_neu

@functools.lru_cache
def loadImagesHDF5(id, scanner):
    file = h5py.File('../../data/segmentation/{}/{}.hdf5'.format(scanner[:3], id), 'r')
    img = np.array(file['img'][:])
    mask = np.array(file['mask'][:])

    center_coord = file.attrs['center_coords']
    center_coord = center_coord.astype(int)
    return img, mask, center_coord

def getImagesHDF5(id, scanner, image_size, aug):
    # file = h5py.File('../../data/segmentation/{}/{}.hdf5'.format(scanner[:3], id), 'r')
    # img = np.array(file['img'][:])
    # mask = np.array(file['mask'][:])

    # center_coord = file.attrs['center_coords']
    # center_coord = center_coord.astype(int)

    img, mask, center_coord = loadImagesHDF5(id, scanner)

    side_length = image_size // 2
    img = img[
        center_coord[0] - side_length: center_coord[0] + side_length,
        center_coord[1] - side_length: center_coord[1] + side_length,
        center_coord[2] - side_length: center_coord[2] + side_length
    ]
    mask = mask[
        center_coord[0] - side_length: center_coord[0] + side_length,
        center_coord[1] - side_length: center_coord[1] + side_length,
        center_coord[2] - side_length: center_coord[2] + side_length
    ]

    if aug:
        flip_axes = np.random.randint(low=0, high=2, size=3)
        for i, i_flip in enumerate(flip_axes):
            if i_flip > 0:
                img = np.flip(img, i)
                mask = np.flip(mask, i)

        rotate_degrees = np.random.randint(low=0, high=4, size=3)
        rotate_degrees = rotate_degrees * 90
        for i, i_degree in enumerate(rotate_degrees):
            img = ndimage.rotate(img, i_degree, axes=(i, (i + 1) % 3))
            mask = ndimage.rotate(mask, i_degree, axes=(i, (i + 1) % 3))

        scaling = np.random.rand()
        scaling = (scaling * 0.2) + 0.9
        img = scaling * img

    return img, mask

class SegmentationDatasetHDF5(Dataset):
    def __init__(self, tuple_list, image_size, aug=False):
        self.tuple_list = tuple_list
        self.image_size = image_size
        self.aug=aug
    
    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, index):
        id = self.tuple_list[index][0]
        scanner = self.tuple_list[index][1]
        img, mask = getImagesHDF5(id, scanner, image_size=self.image_size, aug=self.aug)
        img = torch.tensor(img)
        mask = torch.tensor(mask)
        return img.unsqueeze(0).to(dtype=torch.float), mask.unsqueeze(0).to(dtype=torch.float)

def getDataLoaderHDF5(batch_size, image_size, num_workers, data_ratio=1.0, persistent_workers=False, aug=False):
    tuple_list_alt, tuple_list_neu = getTupleListHDF5()
    train_list_alt, val_list_alt = train_test_split(tuple_list_alt, test_size=0.1, shuffle=False)
    train_list_neu, val_list_neu = train_test_split(tuple_list_neu, test_size=0.1, shuffle=False)
    train_list = train_list_alt + train_list_neu
    val_list = val_list_alt + val_list_neu
    end_ndx_trn = int(len(train_list) * data_ratio)
    train_list = train_list[:end_ndx_trn]
    if not data_ratio == 1.0:
        val_list = val_list[:batch_size]
    train_ds = SegmentationDatasetHDF5(train_list, image_size, aug=aug)
    val_ds = SegmentationDatasetHDF5(val_list, image_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=persistent_workers, shuffle=False)
    return train_dl, val_dl

def getDataLoaderForOverfitting(batch_size, image_size, num_workers, persistent_workers=False, aug=False):
    print('Using overfitting data loader')
    tuple_list_alt, tuple_list_neu = getTupleListHDF5()
    train_list = tuple_list_alt[:5] + tuple_list_neu[:5]
    train_ds = SegmentationDatasetHDF5(train_list, image_size, aug=aug)
    val_ds = SegmentationDatasetHDF5(train_list, image_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, persistent_workers=persistent_workers, shuffle=False)
    return train_dl, val_dl

def getTupleList():
    raw_path_list_alt = glob.glob('../../data/altesCT_Segmentierung/*')
    raw_path_list_alt.sort()
    id_list_alt = [(os.path.split(path)[-1][:-14], 'altesCT') for path in raw_path_list_alt]
    raw_path_list_neu = glob.glob('../../data/neuesCT_Segmentierung/*')
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
    data = nib.load('../../data/{}/{}.nii.gz'.format(scanner_version,id))
    data_mask = nib.load('../../data/{}_Segmentierung/{}-labels.nii.gz'.format(scanner_version, id))
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
    mask = mask > 0
    mask = mask.astype(np.uint8)
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

def getDataLoader(batch_size, image_size, persistent_workers=False):
    tuple_list = getTupleList()
    tuple_list = tuple_list
    train_list, val_list = train_test_split(tuple_list, test_size=0.1, shuffle=False)
    train_ds = SegmentationDataset(train_list, image_size)
    val_ds = SegmentationDataset(val_list, image_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, drop_last=True, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, drop_last=True, persistent_workers=persistent_workers, shuffle=False)
    return train_dl, val_dl