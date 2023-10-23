from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_cosine_lr_with_linear_warmup(optim, warm_up_epochs, T_max, eta_min):
    linear_warm_up = LinearLR(optim, start_factor=1e-8, total_iters=warm_up_epochs)
    cosine_scheduler = CosineAnnealingLR(optim, T_max, eta_min)
    return SequentialLR(optim, schedulers=[linear_warm_up, cosine_scheduler], milestones=[warm_up_epochs])