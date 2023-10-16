import glob
import os
from time import time

import nibabel as nib
import numpy as np
import h5py
from scipy import ndimage

raw_path_list_alt = glob.glob('../../data/altesCT_Segmentierung/*')
raw_path_list_alt.sort()
id_list_alt = [(os.path.split(path)[-1][:-14], 'altesCT') for path in raw_path_list_alt]
raw_path_list_neu = glob.glob('../../data/neuesCT_Segmentierung/*')
raw_path_list_neu.sort()
id_list_neu = [(os.path.split(path)[-1][:-14], 'neuesCT') for path in raw_path_list_neu]
id_list = id_list_alt + id_list_neu

# id_list = [(4238121, 'altesCT')]
f = h5py.File('../../data/segmentationv2.hdf5', 'w')
f.create_dataset(name='img', shape=(len(id_list), 300, 300, 150), chunks=(1, 300, 300, 150), dtype='f')
f.create_dataset(name='mask', shape=(len(id_list), 300, 300, 150), chunks=(1, 300, 300, 150), dtype='i8')
f.attrs['id_list'] = id_list

for ndx, tuple in enumerate(id_list):
    t1 = time()

    # load data
    data = nib.load('../../data/{}/{}.nii.gz'.format(tuple[1], tuple[0]))
    data_mask = nib.load('../../data/{}_Segmentierung/{}-labels.nii.gz'.format(tuple[1], tuple[0]))
    img = data.get_fdata()
    mask = data_mask.get_fdata()

    # calculate auxiliary variables
    zooms = data.header.get_zooms()
    new_zooms = (1, 1, 1)
    new_shape = np.asarray(img.shape) * np.asarray(zooms) / np.asarray(new_zooms)
    interpolation_zoom = np.asarray(zooms) / np.asarray(new_zooms)
    centers = (new_shape // 2).astype(int)

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
    f['img'][ndx] = new_img[centers[0]-150:centers[0]+150, centers[1]-150:centers[1]+150, -150:]
    f['mask'][ndx] = new_mask[centers[0]-150:centers[0]+150, centers[1]-150:centers[1]+150, -150:]
    f.flush()

    t2 = time()
    print(tuple, t2 - t1)
f.close()