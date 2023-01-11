from main import SegmentationTrainingApp
import os

# exp1 = SegmentationTrainingApp(epochs=1000, logdir='HDtest', lr=1e-3, comment='w2lr3HD', loss_fn='XE', XEweight=5)
# exp1.main()

# exp2 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-3, comment='w2lr3HD', loss_fn='XE', XEweight=2)
# exp2.main()

# exp3 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-4, comment='w2lr4HD', loss_fn='XE', XEweight=2)
# exp3.main()

# exp4 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-2, comment='w4lr2HD', loss_fn='XE', XEweight=4)
# exp4.main()

# exp5 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-3, comment='w4lr3HD', loss_fn='XE', XEweight=4)
# exp5.main()

# exp6 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-4, comment='w4lr4HD', loss_fn='XE', XEweight=4)
# exp6.main()

# exp7 = SegmentationTrainingApp(epochs=1000, logdir='dice', lr=1e-2, comment='dicelr2', loss_fn='dice')
# exp7.main()

# exp8 = SegmentationTrainingApp(epochs=1000, logdir='dice', lr=1e-3, comment='dicelr3', loss_fn='dice')
# exp8.main()

# exp9 = SegmentationTrainingApp(epochs=1000, logdir='dice', lr=1e-4, comment='dicelr4', loss_fn='dice')
# exp9.main()

exp9 = SegmentationTrainingApp(epochs=10, logdir='test', lr=1e-3, comment='relposbias', loss_fn='XE', XEweight=5, data_ratio=0.1)
exp9.main()