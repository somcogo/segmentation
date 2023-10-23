import torch
from models.swinunetr import SwinUNETR

def inference(img:torch.Tensor, model, section_size):
    H, W, D = img.shape[-3:]
    pred = torch.zeros((1, 2, H, W, D), device='cuda:0')
    prob = torch.zeros((1, H, W, D), device='cuda:0')
    prob2 = torch.zeros((1, 2, H, W, D), device='cuda:0')
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
                temp1, temp2 = model(patch)
                pred[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp1
                prob2[:, :, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp2
                temp_prob = temp2.argmax(dim=1)
                # if temp_prob.max() > 0:
                #     print(x_c, y_c, z_c)
                prob[:, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp_prob

    pred_class = pred.argmax(dim=1)

    return pred_class, pred, prob, prob2