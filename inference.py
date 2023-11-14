from functools import lru_cache

import torch
import h5py
import numpy as np
from scipy import ndimage

def calculate_dice(pred, mask):
    intersection = (pred * mask).sum(axis=(1, 2, 3))
    dice = (2*intersection)/(pred.sum(axis=(1, 2, 3)) + mask.sum(axis=(1, 2, 3)))
    return dice

def postprocess(pred_class_stack, pred_stack, prob_stack):
    comp_per_img = np.zeros(pred_class_stack.shape[0])
    comp_size_per_img = []
    postprocessed = np.zeros(pred_class_stack.shape)
    postprocessed2 = np.zeros(pred_class_stack.shape)
    postprocessed3 = np.zeros(pred_class_stack.shape)
    for img_ndx, pred_im in enumerate(pred_class_stack):
        labelled_components, num_components = ndimage.label(pred_im)
        comp_per_img[img_ndx] = num_components
        if num_components > 0:
            comp_sizes = np.zeros((num_components))
            comp_sizes2 = np.zeros((num_components))
            comp_sizes3 = np.zeros((num_components))
            for comp in range(1, num_components+1):
                comp_sizes[comp-1] = (labelled_components == comp).sum()
                comp_sizes2[comp-1] = pred_stack[img_ndx,1][labelled_components == comp].sum()
                comp_sizes3[comp-1] = prob_stack[img_ndx,1][labelled_components == comp].sum()
            comp_size_per_img.append(comp_sizes)
            largest_comp = comp_sizes.argmax() + 1
            largest_comp2 = comp_sizes2.argmax() + 1
            largest_comp3 = comp_sizes3.argmax() + 1
            postprocessed[img_ndx] = labelled_components == largest_comp
            postprocessed2[img_ndx] = labelled_components == largest_comp2
            postprocessed3[img_ndx] = labelled_components == largest_comp3

    return comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3

def postprocess_single_img(pred_class, pred, prob):
        labelled_components, num_components = ndimage.label(pred_class)
        postprocessed = np.zeros(pred_class.shape)
        postprocessed2 = np.zeros(pred_class.shape)
        postprocessed3 = np.zeros(pred_class.shape)
        comp_sizes = np.zeros((num_components))
        comp_sizes2 = np.zeros((num_components))
        comp_sizes3 = np.zeros((num_components))
        if num_components > 0:
            for comp in range(1, num_components+1):
                comp_sizes[comp-1] = (labelled_components == comp).sum()
                comp_sizes2[comp-1] = pred[1][labelled_components == comp].sum()
                comp_sizes3[comp-1] = prob[1][labelled_components == comp].sum()
            largest_comp = comp_sizes.argmax() + 1
            largest_comp2 = comp_sizes2.argmax() + 1
            largest_comp3 = comp_sizes3.argmax() + 1
            postprocessed = labelled_components == largest_comp
            postprocessed2 = labelled_components == largest_comp2
            postprocessed3 = labelled_components == largest_comp3

        return num_components, postprocessed, postprocessed2, postprocessed3

# From nnUNet github implementation TODO: don't forget to cite
@lru_cache(maxsize=2)
def compute_gaussian(tile_size, sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device='cuda:0') \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = ndimage.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.to(dtype).to(device)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def inference(img:torch.Tensor, model, section_size, device, gaussian_weights=False):
    model = model.to(device)
    model.eval()
    img = img.to(device)
    H, W, D = img.shape[-3:]
    pred = torch.zeros((1, 2, H, W, D), device=device)
    n_predictions = torch.zeros((H, W, D), device=device)
    gaussian = compute_gaussian(section_size, sigma_scale=1. / 8, device=device) if gaussian_weights else None
    # prob = torch.zeros((1, H, W, D), device=device)
    # prob2 = torch.zeros((1, 2, H, W, D), device=device)
    model.eval()
    x, y, z = section_size
    x_coords = list(range(0, H-x, x // 2))
    x_coords.append(H-x)
    y_coords = list(range(0, W-y, y // 2))
    y_coords.append(W-y)
    z_coords = list(range(0, D-z, z // 2))
    z_coords.append(D-z)
    for x_c in x_coords:
        for y_c in y_coords:
            for z_c in z_coords:
                patch = img[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z]
                temp1 = model(patch)
                pred[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp1.detach() * gaussian if gaussian_weights else temp1.detach()
                n_predictions[x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += gaussian if gaussian_weights else 1
                del temp1
                # prob2[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp2
                # temp_prob = temp2.argmax(dim=1)
                # if temp_prob.max() > 0:
                #     print(x_c, y_c, z_c)
                # prob[:, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp_prob
    pred /= n_predictions
    pred_class = pred.argmax(dim=1)

    return pred_class, pred, torch.nn.functional.softmax(pred, dim=1)

def do_inference_on_val_ds(model, section_size, device, keep_masks=False, log=False, img_number=None):
    inf_file = h5py.File('data/segmentation.hdf5')
    img_ds = inf_file['val']['img']
    mask_ds = inf_file['val']['mask']
    if img_number is None:
        img_number = img_ds.shape[0]

    N, H, W, D = img_ds.shape
    pred_class_stack = np.zeros((img_number, H, W, D))
    pred_stack = np.zeros((img_number, 2, H, W, D))
    prob_stack = np.zeros((img_number, 2, H, W, D))
    for img_index in range(img_number):
        img = torch.from_numpy(np.array(img_ds[img_index])).unsqueeze(0).unsqueeze(0)
        pred_class, pred, prob = inference(img, model, section_size, device)

        pred_class_stack[img_index] = pred_class.detach().cpu()
        pred_stack[img_index] = pred.detach().cpu()
        prob_stack[img_index] = prob.detach().cpu()
        
        if log:
            print(img_index)

    mask = np.array(mask_ds[:img_number])
    comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3 = postprocess(pred_class_stack, pred_stack, prob_stack)
    dice0 = calculate_dice(pred_class_stack, mask)
    dice1 = calculate_dice(postprocessed, mask)
    dice2 = calculate_dice(postprocessed2, mask)
    dice3 = calculate_dice(postprocessed3, mask)

    if keep_masks:
        return dice0, dice1, dice2, dice3, comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3, pred_class_stack, pred_stack, prob_stack
    else:
        return dice0, dice1, dice2, dice3, comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3, pred_stack