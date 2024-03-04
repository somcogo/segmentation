from functools import lru_cache

import torch
import h5py
import numpy as np
from scipy import ndimage

from utils.model_init import model_init

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

    return comp_size_per_img, postprocessed, postprocessed2, postprocessed3

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
def compute_gaussian(tile_size : tuple, sigma_scale: float = 1. / 8,
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

def inference(img:torch.Tensor, model, section_size, device, gaussian_weights=False, overlap=None):
    model = model.to(device)
    model.eval()
    img = img
    H, W, D = img.shape[-3:]
    pred = torch.zeros((1, 2, H, W, D), device='cpu')
    n_predictions = torch.zeros((H, W, D), device='cpu')
    gaussian = compute_gaussian(tile_size=tuple(section_size), device='cpu') if gaussian_weights else None
    # prob = torch.zeros((1, H, W, D), device=device)
    # prob2 = torch.zeros((1, 2, H, W, D), device=device)
    model.eval()
    if overlap is None:
        overlap = 0.5
    x, y, z = section_size
    x_step, y_step, z_step = [int(x*overlap), int(y*overlap), int(z*overlap)]
    x_coords = list(range(0, H-x, x_step))
    x_coords.append(H-x)
    y_coords = list(range(0, W-y, y_step))
    y_coords.append(W-y)
    z_coords = list(range(0, D-z, z_step))
    z_coords.append(D-z)
    for x_c in x_coords:
        for y_c in y_coords:
            for z_c in z_coords:
                patch = img[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z]
                patch = patch.to(device)
                temp1 = model(patch)
                temp1 = temp1.detach().cpu()
                pred[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp1 * gaussian if gaussian_weights else temp1
                n_predictions[x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += gaussian if gaussian_weights else 1
                del temp1, patch
                # prob2[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp2
                # temp_prob = temp2.argmax(dim=1)
                # if temp_prob.max() > 0:
                #     print(x_c, y_c, z_c)
                # prob[:, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp_prob
    pred /= n_predictions
    pred_class = pred.argmax(dim=1)

    return pred_class, pred, torch.nn.functional.softmax(pred, dim=1)

def do_inference_on_val_ds(model, section_size, device, keep_masks=False, log=False, img_number=None, gaussian_weights=False, overlap=None):
    inf_file = h5py.File('/home/hansel/developer/segmentation/data/segmentation.hdf5')
    img_ds = inf_file['val']['img']
    mask_ds = inf_file['val']['mask']
    if img_number is None:
        img_number = [0, img_ds.shape[0]]

    N, H, W, D = img_ds.shape
    pred_class_stack = np.zeros((img_number[1]-img_number[0], H, W, D))
    pred_stack = np.zeros((img_number[1]-img_number[0], 2, H, W, D))
    prob_stack = np.zeros((img_number[1]-img_number[0], 2, H, W, D))
    for img_index in range(img_number[0], img_number[1]):
        img = torch.from_numpy(np.array(img_ds[img_index])).unsqueeze(0).unsqueeze(0)
        pred_class, pred, prob = inference(img, model, section_size, device, gaussian_weights=gaussian_weights, overlap=overlap)

        pred_class_stack[img_index-img_number[0]] = pred_class.detach().cpu()
        pred_stack[img_index-img_number[0]] = pred.detach().cpu()
        prob_stack[img_index-img_number[0]] = prob.detach().cpu()
        
        if log:
            print('scan', img_index-img_number[0], '/', img_number[1]-img_number[0])

    mask = np.array(mask_ds[img_number[0]:img_number[1]])
    comp_size_per_img, postprocessed, postprocessed2, postprocessed3 = postprocess(pred_class_stack, pred_stack, prob_stack)
    dice0 = calculate_dice(pred_class_stack, mask)
    dice1 = calculate_dice(postprocessed, mask)
    dice2 = calculate_dice(postprocessed2, mask)
    dice3 = calculate_dice(postprocessed3, mask)

    if keep_masks:
        return dice0, dice1, dice2, dice3, pred_stack, pred_class_stack, comp_size_per_img, postprocessed, postprocessed2, postprocessed3, 
    else:
        return dice0, dice1, dice2, dice3, comp_size_per_img
    
def do_inference_save_results(save_path, image_size, model=None, model_type=None, model_path=None, device='cuda:0', log=False, img_number=None, gaussian_weights=False, overlap=None):
    if model is None:
        model = model_init(model_type=model_type, image_size=image_size)
        state_dict = torch.load(model_path)['model_state']
        model.load_state_dict(state_dict)

    dice0, dice1, dice2, dice3, pred_stack, pred_class_stack, comp_size_per_img, postprocessed, postprocessed2, postprocessed3 = do_inference_on_val_ds(model, image_size, device, keep_masks=True, log=log, gaussian_weights=gaussian_weights, overlap=overlap, img_number=img_number)
    if log:
        print('Inference done, starting saving')
    data = {'dice':dice0,
            'dice_pp_size':dice1,
            'dice_pp_logit':dice2,
            'dice_pp_prob':dice3,
            'comp_sizes':comp_size_per_img}
    for img_ndx in range(pred_stack.shape[0]):
        data[img_ndx] = pred_stack[img_ndx]
    torch.save(data, save_path)

def analyse_results(inference_dict):
    analysis = {}
    analysis['good_indices'] = []
    analysis['bad_indices'] = []
    analysis['good_indices1'] = []
    analysis['bad_indices1'] = []
    analysis['good_indices2'] = []
    analysis['bad_indices2'] = []
    analysis['good_indices3'] = []
    analysis['bad_indices3'] = []
    img_index = 69
    f = h5py.File('/home/hansel/developer/segmentation/data/segmentation.hdf5', 'r')
    comp_list = np.zeros(72)
    for img_index in range(72):
        mask = f['val']['mask'][img_index]
        pred = inference_dict[img_index]
        prob = np.array(torch.nn.functional.softmax(torch.from_numpy(pred)))
        class_pred = pred.argmax(axis=0)
        num_components, postprocessed, postprocessed2, postprocessed3 = postprocess_single_img(class_pred, pred, prob)
        comp_list[img_index] = num_components if num_components is not None else 0
        if (mask*class_pred).sum()>0:
            analysis['good_indices'].append(img_index)
        else:
            analysis['bad_indices'].append(img_index)
        if (mask*postprocessed).sum()>0:
            analysis['good_indices1'].append(img_index)
        else:
            analysis['bad_indices1'].append(img_index)
        if (mask*postprocessed2).sum()>0:
            analysis['good_indices2'].append(img_index)
        else:
            analysis['bad_indices2'].append(img_index)
        if (mask*postprocessed3).sum()>0:
            analysis['good_indices3'].append(img_index)
        else:
            analysis['bad_indices3'].append(img_index)

    analysis['int_indices'] = []
    for ndx, comp_s in enumerate(comp_list):
        if comp_s != 1:
            analysis['int_indices'].append([ndx, comp_s])
    analysis['comp_sizes'] = []
    for ndx, comp_s in enumerate(inference_dict['comp_sizes']):
        if len(comp_s) > 1:
            analysis['comp_sizes'].append([ndx, comp_s])

    analysis['lower_indices'] = []
    analysis['upper_indices'] = []
    analysis['lower_dice'] = []
    analysis['upper_dice'] = []
    analysis['lower_percentage'] = []
    analysis['upper_percentage'] = []
    analysis['lower_number'] = []
    analysis['upper_number'] = []
    for threshhold in np.linspace(0., 1., num=11, endpoint=True):
        lower_indices = np.where(inference_dict['dice_pp_size']<threshhold)
        analysis['lower_indices'].append(lower_indices)
        analysis['lower_dice'].append(inference_dict['dice_pp_size'][lower_indices])
        analysis['lower_number'].append(len(lower_indices[0]))
        analysis['lower_percentage'].append(len(lower_indices[0])/72)

        upper_indices = np.where(inference_dict['dice_pp_size']>threshhold)
        analysis['upper_indices'].append(upper_indices)
        analysis['upper_dice'].append(inference_dict['dice_pp_size'][upper_indices])
        analysis['upper_number'].append(len(upper_indices[0]))
        analysis['upper_percentage'].append(len(upper_indices[0])/72)

    return analysis

def postprocess_to_single_comp(pred_class):
    if pred_class.sum() > 0:
        labelled_components, num_components = ndimage.label(pred_class)
        comp_sizes = np.zeros((num_components))
        if num_components > 0:
            for comp in range(1, num_components+1):
                comp_sizes[comp-1] = (labelled_components == comp).sum()
            largest_comp = comp_sizes.argmax() + 1
            postprocessed = labelled_components == largest_comp
    else:
        postprocessed = np.zeros_like(pred_class)

    return postprocessed

def threshholds_strict(pred, mask, thresholds):
    prob = np.exp(pred) / np.sum(np.exp(pred), axis=0)[1]
    dices = np.zeros_like(thresholds)
    for i, th in enumerate(thresholds):
        post = postprocess_to_single_comp(prob > th)
        dices[i] = 2*(post * mask).sum()/(post.sum() + mask.sum())

    return dices