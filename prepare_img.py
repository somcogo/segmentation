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


dcm_path = '../data/CTAs/3014684'
dcm_path, _ = check_for_dicomdir(dcm_path)
spacing, _ = get_spacing_px_count(dcm_path)
shape = np.array([*pydicom.dcmread(glob.glob(dcm_path+'/*')[0]).pixel_array.shape, len(glob.glob(dcm_path+'/*'))])
target_spacing = np.array([0.55273438, 0.55273438, 0.5])
vol_from_dcm, sp = load_dcm_into_npy_vol(dcm_path)
nii_file = nib.load('../data/CTAs/3014684-labels.nii.gz')
mask_from_nii = nii_file.get_fdata()
x_orig = np.arange(0, spacing[0]*shape[0], spacing[0])
y_orig = np.arange(0, spacing[1]*shape[1], spacing[1])
z_orig = np.arange(0, spacing[2]*shape[2], spacing[2])
x_nii = np.linspace(x_orig[0], x_orig[-1], 1024)
y_nii = np.linspace(y_orig[0], y_orig[-1], 1024)
z_nii = np.linspace(z_orig[0], z_orig[-1], len(z_orig))
x_targ = np.arange(x_orig[0], x_orig[-1], target_spacing[0])
y_targ = np.arange(y_orig[0], y_orig[-1], target_spacing[1])
z_targ = np.arange(z_orig[0], z_orig[-1], target_spacing[2])
new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(spacing, target_spacing, shape)])
from skimage.transform import resize
im_interp = resize(vol_from_dcm, new_shape, order=3)
im_interp2 = (im_interp-im_interp.min())/(im_interp.max()-im_interp.min())*(vol_from_dcm.max()-vol_from_dcm.min()) + vol_from_dcm.min()
from scipy import interpolate
nearest_interpolator = interpolate.NearestNDInterpolator((x_nii, y_nii, z_nii), mask_from_nii)
targ = np.zeros((len(x_targ), len(y_targ), len(z_targ), 3))
for x_i, x in enumerate(x_targ):
    for y_i, y in enumerate(y_targ):
        for z_i, z in enumerate(z_targ):
            targ[x_i, y_i, z_i] = [x, y, z]
mask_int = nearest_interpolator(targ)
new_aff = [[-target_spacing[0],  0.,  0.,  126.00000], [-0,  target_spacing[1],  0., -274.53125], [ 0., -0.,  target_spacing[2], -368.50000], [ 0.,  0,  0,  1]]
ni5 = nib.Nifti1Image(np.flip(im_interp2.transpose(1, 0, 2), axis=1), new_aff)
# ni2.header.set_zooms((0.2341, 0.2341, 0.5))
nib.save(ni5, '../data/CTAs/3014684-medsp.nii.gz')
new_aff = [[-target_spacing[0],  0.,  0.,  126.00000], [-0,  target_spacing[1],  0., -274.53125], [ 0., -0.,  target_spacing[2], -368.50000], [ 0.,  0,  0,  1]]
ni5 = nib.Nifti1Image(mask_int, new_aff)
# ni2.header.set_zooms((0.2341, 0.2341, 0.5))
nib.save(ni5, '../data/CTAs/3014684-medsp-label.nii.gz')