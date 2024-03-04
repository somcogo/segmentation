import os
import glob

import pydicom
from pydicom.fileset import FileSet
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np

def collect_statistics(dcm_paths):
    spacing = np.zeros((len(dcm_paths), 3))
    pixel_count = np.zeros((len(dcm_paths), 100000))
    for i, path in enumerate(dcm_paths):
        actual_path, img_id = check_for_dicomdir(path)
        sp, px = get_spacing_px_count(actual_path)
        pixel_count[i] = px
        spacing[i] = sp

    return spacing, pixel_count

def get_spacing_px_count(dcm_path):
    slice_paths = glob.glob(os.path.join(dcm_path, '*'))

    px_count = np.zeros(100000)
    spacing = np.zeros(3)
    z_list = []
    for slice_path in slice_paths:
        dcm_ds = pydicom.dcmread(slice_path)
        hu = apply_modality_lut(dcm_ds.pixel_array, dcm_ds).astype(int)
        count = np.bincount(hu.flatten() + 8192)
        px_count[:len(count)] += count
        z_list.append(dcm_ds.SliceLocation)
    z_list.sort()
    spacing[0] = z_list[1] - z_list[0]
    spacing[1:] = dcm_ds.PixelSpacing

    return spacing, px_count

def check_for_dicomdir(dcm_path):
    if 'DICOMDIR' in os.listdir(dcm_path):
        dr, img_id = os.path.split(dcm_path)
        dicomdir_path = os.path.join(dcm_path, 'DICOMDIR')
        ds_dir = pydicom.dcmread(dicomdir_path)
        fs = FileSet(ds_dir)
        root_path = fs.path
        file_ids = fs.find_values("ReferencedFileID")
        actual_path = os.path.join(root_path, *file_ids[0][:-1])
    else:
        actual_path = dcm_path
        img_id = os.path.basename(dcm_path)
    
    return actual_path, img_id

def check_dcm_shape(dcm_path):
    dcm_path, img_id = check_dcm_shape(dcm_path)
    slice_paths = glob.glob(os.path.join())

    dcm_ds = pydicom.dcmread(slice_paths[0])
    hu = apply_modality_lut(dcm_ds.pixel_array, dcm_ds).astype(int)
    
    shape = np.array([*hu.shape, len(slice_paths)])
    return shape

