from glob import glob
import os

import pydicom
from pydicom.fileset import FileSet
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

from collect_stats import get_spacing_px_count, check_for_dicomdir, check_dcm_shape

def prepare_images(dcm_paths, npy_path, stats, interp_order=3):
    for i, dcm_path in enumerate(dcm_paths):
        actual_path, img_id = check_for_dicomdir(dcm_path)

        npy_p = os.path.join(npy_path, img_id + '.npy')
        load_dcm_save_npy(dcm_path=actual_path, npy_path=npy_p, stats=stats, interp_order=interp_order)
        print(img_id, i+1, '/', len(dcm_paths))

def load_dcm_save_npy(dcm_path, npy_path, stats=None, interp_order=3):
    np_volume, old_spacing = load_dcm_into_npy_vol(dcm_path)
    np_volume = preprocess_npy_vol(np_volume, stats, interp_order, old_spacing)

    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, np_volume)

def load_dcm_into_npy_vol(dcm_path):
    slice_loc_slice_list = []
    for path in glob(os.path.join(dcm_path, '*')):
        dcm_ds = pydicom.dcmread(path)
        slice_loc_slice_list.append([dcm_ds.SliceLocation, apply_modality_lut(dcm_ds.pixel_array, dcm_ds).astype(int)])
    slice_loc_slice_list.sort()

    np_volume = np.empty((*slice_loc_slice_list[0][1].shape, len(slice_loc_slice_list)), dtype=int)
    for i, (_, slice_arr) in enumerate(slice_loc_slice_list):
        np_volume[:, :, i] = slice_arr

    spacing = np.array([*dcm_ds.PixelSpacing, slice_loc_slice_list[1][0] - slice_loc_slice_list[0][0]])

    return np_volume, spacing

def preprocess_npy_vol(np_volume, stats, interp_order, old_spacing):

    spacing = stats['spacing']
    if spacing is not None:
        interp_zooms = old_spacing / np.array(spacing)
        np_volume = zoom(np_volume, zoom=interp_zooms, order=interp_order)

    np_volume = np.clip(np_volume, stats['min'], stats['max'])
    np_volume = (np_volume - stats['mean']) / stats['std']

    return np_volume

def prepare_mask(dcm_paths, nii_paths, npy_path, stats, interp_order='nearest'):
    for nii_p in nii_paths:
        img_id = os.path.basename(nii_p)[:7]
        for p in dcm_paths:
            if img_id in os.path.basename(p):
                dcm_p = p
                break
        dcm_p, img_id = check_for_dicomdir(dcm_p)
        
def prepare_single_mask(nii_path, dcm_path, npy_path, stats, interp_order='nearest'):
    orig_spacing = get_spacing_px_count(check_for_dicomdir(dcm_path)[0])
    nii_spacing = np.array([*orig_spacing[:2]*511/1023, orig_spacing[2]])
    new_spacing = stats['spacing']

    nii_file = nib.load()
    dcm_shape = check_dcm_shape(dcm_path)