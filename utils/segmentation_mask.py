import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def draw_segmenation_mask(img:np.ndarray,    # shape (H, W, D)
                          masks:np.ndarray,  # shape (n, H, W, D)
                          colors:np.ndarray, # shape (n, 3)
                          alpha=0.3):
    H, W, D = img.shape
    n, _ = colors.shape
    img = np.expand_dims(img, axis=0)
    img = np.concatenate([img, img, img], axis=0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    masks = masks > 0

    rgb_masks = np.expand_dims(masks, axis=1)
    rgb_masks = np.concatenate([rgb_masks, rgb_masks, rgb_masks], axis=1)
    # rgb_masks shape (n, 3, H, W, D)
    colors = np.broadcast_to(np.expand_dims(colors, axis=(2, 3, 4)), rgb_masks.shape)
    rgb_masks = colors * rgb_masks

    coeffs = np.zeros(rgb_masks.shape)
    rgb_masks_sum = np.zeros(rgb_masks.shape)
    rgb_masks_sum += np.sum(rgb_masks, axis=0, keepdims=True)
    pos_mask = rgb_masks > 0
    coeffs[pos_mask] = (1 - alpha) * rgb_masks[pos_mask] / rgb_masks_sum[pos_mask]
    rgb_masks = rgb_masks * coeffs
    weighted_masks = rgb_masks.sum(dim=0)

    sum_mask = weighted_masks > 0
    img[sum_mask] = img[sum_mask]*alpha + weighted_masks[sum_mask]
    return img.astype(dtype=np.uint8)

def create_animation(img_to_animate):
    fig = plt.figure()
    ims = []
    for image in range(0,img_to_animate.shape[-1]):
        im = plt.imshow(img_to_animate[:,:,:,-image-1].permute(1, 2, 0), 
                        animated=True)
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)
    plt.close()
    html = HTML(ani.to_jshtml())
    html
    return

def animate_seg_mask(img:np.ndarray, masks:np.ndarray, colors:np.ndarray, alpha=0.3):
    img_to_animate = draw_segmenation_mask(img, masks, colors, alpha)
    animation = create_animation(img_to_animate)
    return animation