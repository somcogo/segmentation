from glob import glob
import os

import pydicom
import numpy as np
from scipy.ndimage import zoom

def load_dcm_save_npy(dcm_path, npy_path, spacing=None, interp_order=3):
    slice_loc_slice_list = []
    for path in glob(os.path.join(dcm_path, '*')):
        dcm_ds = pydicom.dcmread(path)
        slice_loc_slice_list.append([dcm_ds.SliceLocation, dcm_ds.pixel_array])
    slice_loc_slice_list.sort()

    np_volume = np.empty((*slice_loc_slice_list[0][1].shape, len(slice_loc_slice_list)), dtype=np.uint16)
    for i, (_, slice_arr) in enumerate(slice_loc_slice_list):
        np_volume[:, :, i] = slice_arr

    if spacing is not None:
        old_spacing = np.array([*dcm_ds.PixelSpacing, slice_loc_slice_list[1][0] - slice_loc_slice_list[0][0]])
        interp_zooms = old_spacing / np.array(spacing)
        np_volume = zoom(np_volume, zoom=interp_zooms, order=interp_order)

    np_volume = (np_volume - np_volume.min()) / (np_volume.max() - np_volume.min())

    # os.makedirs(npy_path, exist_ok=True)
    np.save(npy_path, np_volume)

        