import os

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

from utils.ops import aug_tuple

class SegmentationDataset(Dataset):
    def __init__(self, data_path, mode, aug=False, section='random'):
        super().__init__()
        file = h5py.File(os.path.join(data_path, 'segmentation.hdf5'), 'r')
        self.img_ds = file[mode]['img']
        self.mask_ds = file[mode]['mask']
        self.aug = aug
        self.section = section

    def __len__(self):
        return self.img_ds.shape[0]
    
    def __getitem__(self, index):
        if self.section == 'random':
            rng = np.random.default_rng()
            x1 = rng.integers(0, self.img_ds.shape[1] - 128)
            y1 = rng.integers(0, self.img_ds.shape[2] - 128)
            z1 = rng.integers(0, self.img_ds.shape[3] - 128)
            x2, y2, z2 = x1 + 128, y1 + 128, z1 + 128
        elif self.section == 'whole':
            x1, x2, y1, y2, z1, z2 = 0, 350, 0, 350, 0, 150
        image = np.array(self.img_ds[index, x1:x2, y1:y2, z1:z2])
        mask = np.array(self.mask_ds[index, x1:x2, y1:y2, z1:z2])
        if self.aug:
            image, mask = aug_tuple(image, mask)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0)
    
class SegmentationDatasetTest(Dataset):
    def __init__(self, data_path, mode, aug=False, section='random'):
        super().__init__()
        file = h5py.File(os.path.join(data_path, 'segmentation.hdf5'), 'r')
        self.img_ds = np.array(file[mode]['img'])
        self.mask_ds = np.array(file[mode]['mask'])
        self.aug = aug
        self.section = section

    def __len__(self):
        return self.img_ds.shape[0]
    
    def __getitem__(self, index):
        if self.section == 'random':
            rng = np.random.default_rng()
            x1 = rng.integers(0, self.img_ds.shape[1] - 144)
            y1 = rng.integers(0, self.img_ds.shape[2] - 144)
            z1 = rng.integers(0, self.img_ds.shape[3] - 144)
            x2, y2, z2 = x1 + 144, y1 + 144, z1 + 144
        elif self.section == 'whole':
            x1, x2, y1, y2, z1, z2 = 0, 350, 0, 350, 0, 150
        image = self.img_ds[index, x1:x2, y1:y2, z1:z2]
        mask = self.mask_ds[index, x1:x2, y1:y2, z1:z2]
        if self.aug:
            image, mask = aug_tuple(image, mask)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0)

def getSegmentationDataLoader(batch_size, aug=True, section='random'):
    trn_ds = SegmentationDataset(data_path='data', mode='trn', aug=aug, section=section)
    val_ds = SegmentationDataset(data_path='data', mode='val', aug=False, section=section)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    return trn_dl, val_dl

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
        image = self.data[index]
        mask = self.mask[index]
        img_id = self.id_list[index]
        if self.aug:
            image, mask = aug_tuple(image, mask)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0), img_id
    
def getNewDataLoader(batch_size, persistent_workers=True, aug=True):
    trn_ds = DatasetV2(data_path='data', mode='trn', aug=aug)
    val_ds = DatasetV2(data_path='data', mode='val', aug=False)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
    return trn_dl, val_dl