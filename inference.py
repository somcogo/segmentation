import torch
import h5py
import numpy as np
from scipy import ndimage

from models.swinunetr import SwinUNETR

def calculate_dice(pred, mask):
    intersection = (pred == mask).sum(axis=(-3,-2,-1))
    dice = (2*intersection)/(pred.sum(axis=(-3,-2,-1)) + mask.sum(axis=(-3,-2,-1)))
    return dice

def postprocess(pred_class_stack, pred_stack, prob_stack):
    comp_per_img = np.zeros((72))
    comp_size_per_img = []
    postprocessed = np.zeros((72, 350, 350, 150))
    postprocessed2 = np.zeros((72, 350, 350, 150))
    postprocessed3 = np.zeros((72, 350, 350, 150))
    for img_ndx, pred_im in enumerate(pred_class_stack):
        labelled_components, num_components = ndimage.label(pred_im)
        comp_per_img[img_ndx] = num_components
        comp_sizes = np.zeros((num_components+1))
        comp_sizes2 = np.zeros((num_components+1))
        comp_sizes3 = np.zeros((num_components+1))
        for comp in range(1, num_components+1):
            comp_sizes[comp] = (labelled_components == comp).sum()
            comp_sizes2[comp] = pred_stack[img_ndx,1][labelled_components == comp].sum()
            comp_sizes3[comp] = prob_stack[img_ndx,1][labelled_components == comp].sum()
        comp_size_per_img.append(comp_sizes)
        largest_comp = comp_sizes.argmax()
        largest_comp2 = comp_sizes2.argmax()
        largest_comp3 = comp_sizes3.argmax()
        postprocessed[img_ndx] = labelled_components == largest_comp
        postprocessed2[img_ndx] = labelled_components == largest_comp2
        postprocessed3[img_ndx] = labelled_components == largest_comp3

    return comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3

def inference(img:torch.Tensor, model, section_size, device):
    model = model.to(device)
    model.eval()
    img = img.to(device)
    H, W, D = img.shape[-3:]
    pred = torch.zeros((1, 2, H, W, D), device=device)
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
                pred[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp1.detach()
                del temp1
                # prob2[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp2
                # temp_prob = temp2.argmax(dim=1)
                # if temp_prob.max() > 0:
                #     print(x_c, y_c, z_c)
                # prob[:, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp_prob

    pred_class = pred.argmax(dim=1)

    return pred_class, pred, torch.nn.functional.softmax(pred, dim=1)

def do_inference_on_val_ds(model, section_size, device, keep_masks=False, log=False):
    inf_file = h5py.File('../data/segmentation.hdf5')
    img_ds = inf_file['val']['img']
    mask_ds = inf_file['val']['mask']

    shape = [*img_ds.shape]
    pred_class_stack = np.zeros(shape)
    shape.insert(1, 2)
    pred_stack = np.zeros(shape)
    prob_stack = np.zeros(shape)
    for img_index in range(img_ds.shape[0]):
        img = torch.from_numpy(np.array(img_ds[img_index])).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(np.array(mask_ds[img_index])).unsqueeze(0)
        pred_class, pred, prob = inference(img, model, section_size, device)
        pred_class = pred_class.detach().cpu()
        pred = pred.detach().cpu()
        prob = prob.detach().cpu()

        if keep_masks:
            pred_class_stack[img_index] = pred_class
            pred_stack[img_index] = pred
            prob_stack[img_index] = prob
        
        if log:
            print(img_index)

    comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3 = postprocess(pred_class_stack, pred_stack, prob_stack)
    dice0 = calculate_dice(pred_class, mask)
    dice1 = calculate_dice(postprocessed, mask)
    dice2 = calculate_dice(postprocessed2, mask)
    dice3 = calculate_dice(postprocessed3, mask)

    if keep_masks:
        return dice0, dice1, dice2, dice3, comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3, pred_class_stack, pred_stack, prob_stack
    else:
        return dice0, dice1, dice2, dice3, comp_size_per_img, postprocessed, comp_sizes, postprocessed2, comp_sizes2, postprocessed3, comp_sizes3