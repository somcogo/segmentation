import torch

def draw_segmenation_mask(img, masks, colors, device, alpha=0.3):
    rgb_masks = torch.concat([masks.unsqueeze(1), masks.unsqueeze(1), masks.unsqueeze(1)], dim=1)
    colors = colors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
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