import glob
import os
from time import time

import nibabel as nib
import numpy as np
import h5py
from scipy import ndimage
import torch


id_list = torch.load('../data/asbach_id_data.pt')
size_data = torch.load('../data/segmentation_size_data.pt')
f = h5py.File('../data/segmentation_original_val_asbach.hdf5', 'a')

for key in ['val']:
    if key not in f.keys():
        group = f.create_group(key)
    else:
        group = f[key]
    for site in ['asbach']:
        if site not in group.keys():
            site_group = group.create_group(site)
        else:
            site_group = group[site]
        site_group.attrs['id_list'] = id_list[key]
        for ndx, img_id in enumerate(id_list[key]):
            t1 = time()

            # load data
            data = nib.load('../data/segmentation_CT/{}/{}.nii'.format(site, img_id))
            data_mask = nib.load('../data/segmentation_CT/{}_labels/{}-labels.nii.gz'.format(site, img_id))
            img = data.get_fdata()
            mask = data_mask.get_fdata()

            # calculate auxiliary variables
            zooms = data.header.get_zooms()
            new_zooms = (1, 1, 1)
            # new_shape = np.asarray(img.shape) * np.asarray(zooms) / np.asarray(new_zooms)
            # new_shape = new_shape.astype(np.int32)
            interpolation_zoom = np.asarray(zooms) / np.asarray(new_zooms)

            # preprocessing
            new_img = ndimage.zoom(img, interpolation_zoom, order=1)
            low_perc = np.percentile(new_img, 2.5)
            high_perc = np.percentile(new_img, 97.5)
            if high_perc == low_perc:
                new_img = new_img / (2 * high_perc)
            else:
                new_img = (new_img - low_perc) / (high_perc - low_perc)
            new_mask = ndimage.zoom(mask, interpolation_zoom, order=1)
            new_mask = new_mask > 0.1
            new_mask = new_mask.astype(np.uint8)
            new_shape = new_img.shape

            # create datasets and attributes
            site_group.create_dataset(name='{}'.format(img_id), data=new_img, dtype='f')
            site_group.create_dataset(name='{}_mask'.format(img_id), data=new_mask, dtype='i8')

            t2 = time()
            print(img_id, key, site, t2 - t1)
f.close()