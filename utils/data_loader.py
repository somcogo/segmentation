import os
import glob

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

from utils.ops import crop_section_and_aug

class NewSegmentationDataset(Dataset):
    def __init__(self, data_path, mode, aug='nnunet', section='random', image_size=None, foreground_pref_chance=0., rng_seed=None):
        super().__init__()
        self.data_path = data_path
        self.img_ids = [os.path.basename(p).split('.')[0] for p in glob.glob(data_path)]
        self.img_ids.sort()
        self.aug = aug
        self.section = section
        self.image_size = image_size
        self.foreground_pref_chance = foreground_pref_chance
        self.mode = mode
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        image = np.load(os.path.join(self.data_path, 'img', self.img_ids[index] + '.npy'))
        try:
            mask = np.load(os.path.join(self.data_path, 'mask', self.img_ids[index] + '-labels.npy'))
        except:
            mask = np.zeros_like(image)
        image, mask = crop_section_and_aug(image, mask, self.image_size, mode=self.mode, aug=self.aug, foreground_pref_chance=self.foreground_pref_chance, rng_seed=self.rng_seed)
        return image.unsqueeze(0), mask.unsqueeze(0)

class NewSwarmSegmentationDataset(Dataset):
    def __init__(self, data_path, mode, site, aug='nnunet', section='random', image_size=None, foreground_pref_chance=0., rng_seed=None):
        super().__init__()
        self.data_path = data_path
        self.img_ids = torch.load('data/seg_retrain_id_data.pt')[mode][site]
        self.img_ids.sort()
        self.aug = aug
        self.section = section
        self.image_size = image_size
        self.foreground_pref_chance = foreground_pref_chance
        self.site = site
        self.mode = mode
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        image = np.load(os.path.join(self.data_path, 'img', self.img_ids[index] + '.npy'))
        try:
            mask = np.load(os.path.join(self.data_path, 'mask', self.img_ids[index] + '-labels.npy'))
        except:
            mask = np.zeros_like(image)
        image, mask = crop_section_and_aug(image, mask, self.image_size, mode=self.mode, aug=self.aug, foreground_pref_chance=self.foreground_pref_chance, rng_seed=self.rng_seed)
        return image.unsqueeze(0), mask.unsqueeze(0)

class SegmentationDataset(Dataset):
    def __init__(self, data_path, mode, aug='nnunet', section='random', image_size=None, foreground_pref_chance=0., rng_seed=None):
        super().__init__()
        file = h5py.File(os.path.join(data_path, 'segmentation.hdf5'), 'r')
        self.img_ds = file[mode]['img']
        self.mask_ds = file[mode]['mask']
        self.aug = aug
        self.section = section
        self.image_size = image_size
        self.foreground_pref_chance = foreground_pref_chance
        self.mode = mode
        self.rng_seed = rng_seed

    def __len__(self):
        return self.img_ds.shape[0]
    
    def __getitem__(self, index):
        image = np.array(self.img_ds[index])
        mask = np.array(self.mask_ds[index])
        image, mask = crop_section_and_aug(image, mask, self.image_size, mode=self.mode, aug=self.aug, foreground_pref_chance=self.foreground_pref_chance, rng_seed=self.rng_seed)
        return image.unsqueeze(0), mask.unsqueeze(0)

class SwarmSegmentationDataset(Dataset):
    def __init__(self, data_path, mode, site, aug='nnunet', section='random', image_size=None, foreground_pref_chance=0., rng_seed=None):
        super().__init__()
        self.id_dict = torch.load('/home/hansel/developer/segmentation/data/segmentation_id_data.pt')[mode]
        file = h5py.File(os.path.join(data_path, 'segmentation.hdf5'), 'r')
        self.img_ds = file[mode]['img']
        self.mask_ds = file[mode]['mask']
        self.aug = aug
        self.section = section
        self.image_size = image_size
        self.foreground_pref_chance = foreground_pref_chance
        self.site = site
        self.mode = mode
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.id_dict[self.site])
    
    def __getitem__(self, index):
        if self.site == 'neu':
            index = index + len(self.id_dict['alt'])
        elif self.site == 'asbach':
            index = index + len(self.id_dict['alt']) + len(self.id_dict['neu'])

        image = np.array(self.img_ds[index])
        mask = np.array(self.mask_ds[index])
        image, mask = crop_section_and_aug(image, mask, self.image_size, mode=self.mode, aug=self.aug, foreground_pref_chance=self.foreground_pref_chance, rng_seed=self.rng_seed)
        return image.unsqueeze(0), mask.unsqueeze(0)

def getSegmentationDataLoader(batch_size, aug='nnunet', section='random', image_size=(64, 64, 64), foreground_pref_chance=0., dataset='new'):
    ds_class = SegmentationDataset if dataset != 'new' else NewSegmentationDataset
    trn_ds = ds_class(data_path='/home/hansel/developer/segmentation/data', mode='trn', aug=aug, section=section, image_size=image_size, foreground_pref_chance=foreground_pref_chance)
    val_ds = ds_class(data_path='/home/hansel/developer/segmentation/data', mode='val', aug=aug, section=section, image_size=image_size, foreground_pref_chance=foreground_pref_chance)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return trn_dl, val_dl

def getSwarmSegmentationDataLoader(batch_size, aug=True, section='random', image_size=(64, 64, 64), foreground_pref_chance=0., dataset='new'):
    ds_class = SwarmSegmentationDataset if dataset != 'new' else NewSwarmSegmentationDataset
    trn_ds_list = [ds_class('/home/hansel/developer/segmentation/data', 'trn', site, aug, section, image_size=image_size, foreground_pref_chance=foreground_pref_chance) for site in ['alt', 'neu', 'asbach']]
    val_ds_list = [ds_class('/home/hansel/developer/segmentation/data', 'val', site, False, section, image_size=image_size, foreground_pref_chance=foreground_pref_chance) for site in ['alt', 'neu', 'asbach']]
    trn_dl_list = [DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True) for val_ds in val_ds_list]
    return trn_dl_list, val_dl_list

# class DatasetV2(Dataset):
#     def __init__(self, data_path, mode, aug=False, section='large'):
#         super().__init__()
#         file = h5py.File(os.path.join(data_path, 'segmentation_{}.hdf5'.format(mode)), 'r')
#         self.mode = mode
#         self.section = section
#         self.img_ds = file['img']
#         self.mask_ds = file['mask']
#         self.id_list = file.attrs['id_list']
#         self.aug = aug

#     def __len__(self):
#         return len(self.id_list)
    
#     def __getitem__(self, index):
#         image = self.data[index]
#         mask = self.mask[index]
#         img_id = self.id_list[index]
#         if self.aug:
#             image, mask = aug_tuple(image, mask)
#         return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0), img_id
    
# def getNewDataLoader(batch_size, persistent_workers=True, aug=True):
#     trn_ds = DatasetV2(data_path='data', mode='trn', aug=aug)
#     val_ds = DatasetV2(data_path='data', mode='val', aug=False)
#     trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
#     val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=persistent_workers)
#     return trn_dl, val_dl