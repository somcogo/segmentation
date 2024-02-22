import os

import nibabel as nib
import h5py
import numpy as np
import torch

from inference import postprocess_single_img
from utils.segmentation_mask import draw_segmenation_mask
from utils.nii2dmc import nifti2dicom_1file

def output_to_dcm_single_image(path_to_results, nifti_path, dcm_path, img_ndx, inf_result, rgb=False, affine=None):
    f = h5py.File('data/segmentation.hdf5', 'r')
    img = f['val']['img'][img_ndx]

    # inf_result = torch.load(path_to_results)
    pred = inf_result[img_ndx]
    prob = np.array(torch.nn.functional.softmax(torch.from_numpy(pred)))
    class_pred = pred.argmax(axis=0)
    num_components, postprocessed, postprocessed2, postprocessed3 = postprocess_single_img(class_pred, pred, prob)

    id_data = torch.load('/home/somahansel/developer/segmentation/data/segmentation_id_data.pt')
    if img_ndx > 62:
        img_id = id_data['val']['asbach'][img_ndx-63]
    elif img_ndx > 33:
        img_id = id_data['val']['neu'][img_ndx-34]
    else:
        img_id = id_data['val']['alt'][img_ndx]

    if affine is None:
        nifti_f = nib.load('../data/segmentation_CT/alt/3014684.nii')
        affine = nifti_f.affine

    img_nifti = nib.Nifti1Image(img, affine)
    img_nifti.header.set_zooms((0.46875, 0.46875, 1.0))
    os.makedirs(nifti_path, exist_ok=True)
    os.makedirs(dcm_path, exist_ok=True)
    nib.save(img_nifti, os.path.join(nifti_path, '{}.nii.gz'.format(img_id)))
    nifti2dicom_1file(os.path.join(nifti_path, '{}.nii.gz'.format(img_id)),
                      os.path.join(dcm_path, '{}'.format(img_id)))

    messages = []
    if postprocessed.sum() > 0:
        seg_mask = np.concatenate([np.expand_dims(postprocessed, axis=0)], axis=0)
        x_max = np.where(postprocessed)[0][np.where(np.where(postprocessed)[0] == np.where(postprocessed)[0].max())[0][0]]
        x_min = np.where(postprocessed)[0][np.where(np.where(postprocessed)[0] == np.where(postprocessed)[0].min())[0][0]]
        if x_max > 175 and x_min < 175:
            messages.append('segmentation on both sides ' + str(img_ndx) + ' ' + str(img_id))
        elif x_max > 160 and x_max < 190:
            messages.append('ambiguous side selection ' + str(img_ndx) + ' ' + str(img_id))
        if x_max > 170:
            where_index = np.where(np.where(postprocessed)[0] == np.where(postprocessed)[0].max())[0][0]
        else:
            where_index = np.where(np.where(postprocessed)[0] == np.where(postprocessed)[0].min())[0][0]
        x, y, z = np.where(postprocessed)[0][where_index], np.where(postprocessed)[1][where_index], np.where(postprocessed)[2][where_index]
        value = 30
        seg_mask[:, x-value, y-value:y+value, max(z-value//2, 0):z+value//2] = 1
        seg_mask[:, x+value, y-value:y+value, max(z-value//2, 0):z+value//2] = 1
        seg_mask[:, x-value:x+value, y-value, max(z-value//2, 0):z+value//2] = 1
        seg_mask[:, x-value:x+value, y+value, max(z-value//2, 0):z+value//2] = 1
        seg_mask[:, x-value:x+value, y-value:y+value, max(z-value//2, 0)] = 1
        seg_mask[:, x-value:x+value, y-value:y+value, z+value//2] = 1
        seg_img = draw_segmenation_mask(img, seg_mask, np.array([[255, 10, 10]]))

        seg_nifti = nib.Nifti1Image(seg_img.transpose(1,2,3,0), affine)
        seg_nifti.header.set_zooms((0.46875, 0.46875, 1.0, 1.0))
        nib.save(seg_nifti, os.path.join(nifti_path, '{}_seg.nii.gz'.format(img_id)))
        nifti2dicom_1file(os.path.join(nifti_path, '{}_seg.nii.gz'.format(img_id)),
                          os.path.join(dcm_path, '{}_seg'.format(img_id)))
    else:
        messages.append('empty prediction ' + str(img_ndx) + ' ' + str(img_id))
    return messages

def output_to_dcm_ds(path_to_results, nifti_path, dcm_path, affine=None):
    inf_result = torch.load(path_to_results)
    messages = []
    for img_ndx in range(72):
        m = output_to_dcm_single_image(path_to_results=path_to_results, nifti_path=nifti_path, dcm_path=dcm_path, img_ndx=img_ndx, affine=affine, inf_result=inf_result)
        messages = messages + m
    for mes in messages:
        print(mes)
    