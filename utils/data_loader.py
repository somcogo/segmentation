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
    trn_ds = SegmentationDataset(data_path='../../data', mode='trn', aug=aug, section=section)
    val_ds = SegmentationDataset(data_path='../../data', mode='val', aug=False, section=section)
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
        if self.section == 'large':
            x1, x2, y1, y2, z1, z2 = [6, 294, 6, 294, 0, 144]
        elif self.section == 'random':
            rng = np.random.default_rng()
            x1 = rng.integers(0, 172)
            y1 = rng.integers(0, 172)
            z1 = rng.integers(0, 22)
            x2 = x1 + 128
            y2 = y1 + 128
            z2 = z1 + 128
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