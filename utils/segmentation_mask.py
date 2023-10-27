import torch

def draw_segmenation_mask(img, masks, colors, device, alpha=0.3):
    img = torch.tensor(img.copy()).unsqueeze(0)
    img = torch.concat([img, img, img], dim=0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.to(dtype=torch.uint8)
    masks = torch.tensor(masks.copy()) > 0

    rgb_masks = torch.concat([masks.unsqueeze(1), masks.unsqueeze(1), masks.unsqueeze(1)], dim=1)
    colors = torch.tensor(colors).expand(rgb_masks.shape)
    rgb_masks = colors * rgb_masks

    coeffs = torch.zeros(rgb_masks.shape, device=device)
    rgb_masks_sum = torch.zeros(rgb_masks.shape, device=device)
    rgb_masks_sum += torch.sum(rgb_masks, dim=0, keepdim=True)
    pos_mask = rgb_masks > 0
    coeffs[pos_mask] = (1 - alpha) * rgb_masks[pos_mask] / rgb_masks_sum[pos_mask]
    rgb_masks = rgb_masks * coeffs
    weighted_masks = rgb_masks.sum(dim=0)

    img = img.to(dtype=torch.float)
    sum_mask = weighted_masks > 0
    img[sum_mask] = img[sum_mask]*alpha + weighted_masks[sum_mask]
    return img.to(dtype=torch.uint8)