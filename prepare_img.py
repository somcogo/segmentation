from glob import glob
import os
import time

import pydicom
from pydicom.fileset import FileSet
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from skimage.transform import resize
from scipy import interpolate

from collect_stats import get_spacing_px_count, check_for_dicomdir, check_dcm_shape

def prepare_images(dcm_paths, nii_dir, image_dir, mask_dir, stats, interp_order=3):
    for i, dcm_path in enumerate(dcm_paths):
        t1 = time.time()
        actual_path, img_id = check_for_dicomdir(dcm_path)
        nii_path = os.path.join(nii_dir, img_id + '-labels.nii.gz') if nii_dir is not None else None
        if nii_path is not None and not os.path.isfile(nii_path):
            continue
        image_path = os.path.join(image_dir, img_id + '.npy')
        mask_path = os.path.join(mask_dir, img_id + '-labels.npy') if mask_dir is not None else None

        load_dcm_save_npy(dcm_path=actual_path, nii_path=nii_path, image_path=image_path, mask_path=mask_path, stats=stats, interp_order=interp_order)
        t2 = time.time()
        print(img_id, i+1, '/', len(dcm_paths), t2 - t1)

def load_dcm_save_npy(dcm_path, nii_path, image_path, mask_path, stats=None, interp_order=3):
    image, mask = preprocess_single_image(dcm_path, nii_path, stats, interp_order)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    np.save(image_path, image)
    if mask is not None:
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        np.save(mask_path, mask)

def preprocess_single_image(dcm_path, mask_path, stats, img_interp_order=3):
    dcm_path, img_id = check_for_dicomdir(dcm_path)
    vol_from_dcm, orig_spacing = load_dcm_into_npy_vol(dcm_path)
    orig_shape = vol_from_dcm.shape

    target_spacing = stats['spacing']
    final_shape = np.array([int(round(i / j * k)) for i, j, k in zip(orig_spacing, target_spacing, orig_shape)])
    final_img = resize(vol_from_dcm, final_shape, order=img_interp_order, preserve_range=True)

    final_img = np.clip(final_img, a_min=stats['min'], a_max=stats['max'])
    final_img = (final_img - stats['mean']) / stats['std']

    if mask_path is not None:
        mask_file = nib.load(mask_path)
        mask_from_nii = mask_file.get_fdata()
        mask_from_nii = mask_from_nii > 0

        final_mask = resize(mask_from_nii, final_shape, order=0)
        final_mask = np.flip(final_mask, axis=1).transpose(1, 0, 2)
    else:
        final_mask = None

    return final_img, final_mask

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



# dcm_path = '../data/CTAs/3014684'
# dcm_path, _ = check_for_dicomdir(dcm_path)
# spacing, _ = get_spacing_px_count(dcm_path)
# shape = np.array([*pydicom.dcmread(glob.glob(dcm_path+'/*')[0]).pixel_array.shape, len(glob.glob(dcm_path+'/*'))])
# target_spacing = np.array([0.55273438, 0.55273438, 0.5])
# vol_from_dcm, sp = load_dcm_into_npy_vol(dcm_path)
# # nii_file = nib.load('../data/CTAs/3014684-labels.nii.gz')
# # mask_from_nii = nii_file.get_fdata()
# x_orig = np.arange(0, spacing[0]*shape[0], spacing[0])
# y_orig = np.arange(0, spacing[1]*shape[1], spacing[1])
# z_orig = np.arange(0, spacing[2]*shape[2], spacing[2])
# x_nii = np.linspace(x_orig[0], x_orig[-1], 1024)
# y_nii = np.linspace(y_orig[0], y_orig[-1], 1024)
# z_nii = np.linspace(z_orig[0], z_orig[-1], len(z_orig))
# x_targ = np.arange(x_orig[0], x_orig[-1], target_spacing[0])
# y_targ = np.arange(y_orig[0], y_orig[-1], target_spacing[1])
# z_targ = np.arange(z_orig[0], z_orig[-1], target_spacing[2])
# new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(spacing, target_spacing, shape)])
# im_interp = resize(vol_from_dcm, new_shape, order=3)
# im_interp2 = (im_interp-im_interp.min())/(im_interp.max()-im_interp.min())*(vol_from_dcm.max()-vol_from_dcm.min()) + vol_from_dcm.min()
# nearest_interpolator = interpolate.NearestNDInterpolator((x_nii, y_nii, z_nii), mask_from_nii)
# targ = np.zeros((len(x_targ), len(y_targ), len(z_targ), 3))
# for x_i, x in enumerate(x_targ):
#     for y_i, y in enumerate(y_targ):
#         for z_i, z in enumerate(z_targ):
#             targ[x_i, y_i, z_i] = [x, y, z]
# mask_int = nearest_interpolator(targ)
# new_aff = [[-target_spacing[0],  0.,  0.,  126.00000], [-0,  target_spacing[1],  0., -274.53125], [ 0., -0.,  target_spacing[2], -368.50000], [ 0.,  0,  0,  1]]
# ni5 = nib.Nifti1Image(np.flip(im_interp2.transpose(1, 0, 2), axis=1), new_aff)
# # ni2.header.set_zooms((0.2341, 0.2341, 0.5))
# # nib.save(ni5, '../data/CTAs/3014684-medsp.nii.gz')
# new_aff = [[-target_spacing[0],  0.,  0.,  126.00000], [-0,  target_spacing[1],  0., -274.53125], [ 0., -0.,  target_spacing[2], -368.50000], [ 0.,  0,  0,  1]]
# ni5 = nib.Nifti1Image(mask_int, new_aff)
# # ni2.header.set_zooms((0.2341, 0.2341, 0.5))
# # nib.save(ni5, '../data/CTAs/3014684-medsp-label.nii.gz')
