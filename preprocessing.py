import numpy as np
import nibabel as nib
import h5py
from scipy import ndimage

import glob
import os
from time import time

from utils.data_loader import getCenterCoords

# raw_path_list_alt = glob.glob('../altesCT_Segmentierung/*')
# raw_path_list_alt.sort()
# id_list_alt = [(os.path.split(path)[-1][:-14], 'altesCT') for path in raw_path_list_alt]
# raw_path_list_neu = glob.glob('../neuesCT_Segmentierung/*')
# raw_path_list_neu.sort()
# id_list_neu = [(os.path.split(path)[-1][:-14], 'neuesCT') for path in raw_path_list_neu]
# id_list = id_list_alt + id_list_neu

id_list = [(4238121, 'altesCT')]

for tuple in id_list:
    t1 = time()

    # load data
    data = nib.load('../{}/{}.nii.gz'.format(tuple[1], tuple[0]))
    data_mask = nib.load('../{}_Segmentierung/{}-labels.nii.gz'.format(tuple[1], tuple[0]))
    img = data.get_fdata()
    mask = data_mask.get_fdata()

    # calculate auxiliary variables
    zooms = data.header.get_zooms()
    new_zooms = (0.9375, 0.9375, 0.9375)
    new_shape = np.asarray(img.shape) * np.asarray(zooms) / np.asarray(new_zooms)
    interpolation_zoom = np.asarray(zooms) / np.asarray(new_zooms)
    center_coords = getCenterCoords(mask)
    new_center_coords = center_coords * interpolation_zoom

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

    # create datasets and attributes
    f = h5py.File('{}/{}.hdf5'.format(tuple[1][0:3],tuple[0]), 'w')
    f.create_dataset('img', data=new_img, chunks=True)
    f.create_dataset('mask', data=new_mask, chunks=True)
    f.attrs['patient_id'] = tuple[0]
    f.attrs['scanner'] = tuple[1]
    f.attrs['mm/px'] = 0.9375
    f.attrs['center_coords'] = new_center_coords

    t2 = time()
    print(tuple, t2 - t1)

    # saving nifti
    # print('saving nifti')
    # nifti = nib.Nifti1Image(new_img, np.eye(4))
    # nifti_mask = nib.Nifti1Image(new_mask, np.eye(4))
    # nib.save(nifti, '{}.nii.gz'.format(tuple[0]))
    # nib.save(nifti_mask, '{}_mask.nii.gz'.format(tuple[0]))
    # print(new_center_coords)
    # break