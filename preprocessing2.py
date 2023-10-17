import glob
import os
import time

import h5py
import numpy as np
import torch
from scipy import ndimage

f = h5py.File('../data/segmentation.hdf5', 'a')
id_data = torch.load('../data/segmentation_id_data.pt')

for mode in ['trn', 'val']:
    mode_group = f.create_group(mode)
    scan_count = sum([len(site_list) for site_list in id_data[mode].values()])
    mode_group.create_dataset('img', shape=(scan_count, 350, 350, 150), chunks=(1, 350, 350, 150), dtype='f')
    mode_group.create_dataset('mask',  shape=(scan_count, 350, 350, 150), chunks=(1, 350, 350, 150), dtype='i8')

    for site in ['alt', 'neu', 'asbach']:
        old_file = h5py.File('../data/segmentation_original_{}_{}.hdf5'.format(mode, site), 'a')
        for ndx, img_id in enumerate(id_data[mode][site]):
            t1 = time.time()
            if site == 'neu':
                ndx = ndx + len(id_data[mode]['alt'])
            elif site == 'asbach':
                ndx = ndx + len(id_data[mode]['alt']) + len(id_data[mode]['neu'])
            old_img = np.array(old_file[mode][site][str(img_id)])
            old_mask = np.array(old_file[mode][site][str(img_id)+'_mask'])
            center = ndimage.center_of_mass(old_img[:,:,-100])
            center = (int(center[0]), int(center[1]))

            new_img = old_img[center[0]-175:center[0]+175, center[1]-150:center[1]+200, -150:]
            x_pad, y_pad, z_pad = 350-new_img.shape[0], 350-new_img.shape[1], 150-new_img.shape[2]
            new_img = np.pad(new_img, ((0, x_pad), (0, y_pad), (0, z_pad)))
            mode_group['img'][ndx] = new_img

            new_mask = old_mask[center[0]-175:center[0]+175, center[1]-150:center[1]+200, -150:]
            x_pad, y_pad, z_pad = 350-new_mask.shape[0], 350-new_mask.shape[1], 150-new_mask.shape[2]
            new_mask = np.pad(new_mask, ((0, x_pad), (0, y_pad), (0, z_pad)))
            mode_group['mask'][ndx] = new_mask
            t2 = time.time()
            print(mode, site, img_id, ndx, '/', scan_count, t2-t1)
        old_file.close()