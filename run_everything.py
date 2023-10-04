from main import SegmentationTrainingApp
import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# torch.set_num_threads(8)
# exp1 = SegmentationTrainingApp(epochs=10000, logdir='type2', lr=1e-3, comment='w2lr3Wt2drAUGs2', loss_fn='XE', XEweight=2, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True, scheduler_type='cosinewarmre', batch_size=32)
# exp1.main()

# exp2 = SegmentationTrainingApp(epochs=10000, logdir='type2', lr=1e-3, comment='w4lr3Wt2drAUGs2', loss_fn='XE', XEweight=4, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True, scheduler_type='cosinewarmre', batch_size=32)
# exp2.main()

# exp3 = SegmentationTrainingApp(epochs=10000, logdir='test', lr=1e-3, comment='debugging', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True, scheduler_type='cosinewarmre', batch_size=32)
# exp3.main()
exp3 = SegmentationTrainingApp(epochs=10000, logdir='performancecheck', lr=1e-3, batch_size=4, comment='e10000-b4-lr3-lfXE-XEweight-8-adamw-swinttpye2-droprate0.1-augTrue-cosinewarmre-newDL', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True, scheduler_type='cosinewarmre')
exp3.main()

# exp4 = SegmentationTrainingApp(epochs=10000, logdir='type2', lr=1e-3, comment='w16lr3Wt2drAUGs2', loss_fn='XE', XEweight=16, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True, scheduler_type='cosinewarmre', batch_size=32)
# exp4.main()

# exp5 = SegmentationTrainingApp(epochs=250, logdir='type2', lr=1e-3, comment='w8lr3Wt2dr', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, )
# exp5.main()

# exp6 = SegmentationTrainingApp(epochs=250, logdir='type2', lr=1e-3, comment='w8lr3Wt2draug', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, aug=True)
# exp6.main()

# exp7 = SegmentationTrainingApp(epochs=250, logdir='type2', lr=1e-3, comment='w8lr3Wt2dradr', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, attn_drop_rate=0.1, )
# exp7.main()

# exp8 = SegmentationTrainingApp(epochs=250, logdir='type2', lr=1e-3, comment='w8lr3Wt2dradraug', loss_fn='XE', XEweight=8, optimizer_type='adamw', weight_decay=1e-5, swin_type=2, drop_rate=0.1, attn_drop_rate=0.1, aug=True)
# exp8.main()

# exp9 = SegmentationTrainingApp(epochs=1000, logdir='dice', lr=1e-4, comment='dicelr4', loss_fn='dice')
# exp9.main()

# exp9 = SegmentationTrainingApp(epochs=10, logdir='test', lr=1e-3, comment='aug', loss_fn='XE', XEweight=3, data_ratio=0.1, overfitting=True,  optimizer_type='adamw', weight_decay=1e-5, swin_type=3, aug=True)
# exp9.main()