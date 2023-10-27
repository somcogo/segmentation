import torch
import h5py
import numpy as np

from models.swinunetr import SwinUNETR

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
                print(x_c, y_c, z_c)
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
    inf_file = h5py.File('../../data/segmentation.hdf5')
    img_ds = inf_file['val']['img']
    mask_ds = inf_file['val']['mask']

    dice_scores = np.zeros((img_ds.shape[0]))
    if keep_masks:
        pred_class_stack = np.zeros((img_ds.shape))
        pred_stack = np.zeros(([*img_ds.shape].insert(1, 2)))
        prob_stack = np.zeros(([*img_ds.shape].insert(1, 2)))
    for img_index in range(img_ds.shape[0]):
        img = torch.from_numpy(np.array(img_ds[img_index]))
        mask = torch.from_numpy(np.array(mask_ds[img_index]))
        pred_class, pred, prob = inference(img, model, section_size, device)

        intersect = (pred_class * mask).sum(axis=(1, 2, 3))
        dice_scores[img_index] = intersect/(pred_class.sum(axis=(1, 2, 3)) + mask.sum(axis=(1, 2, 3)))

        if keep_masks:
            pred_class_stack[img_index] = pred_class
            pred_stack[img_index] = pred
            prob_stack[img_index] = prob
        
        if log:
            print(img_index)

    if keep_masks:
        return dice_scores, pred_class_stack, pred_stack, prob_stack
    else:
        return dice_scores