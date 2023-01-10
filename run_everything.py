from main import SegmentationTrainingApp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

exp1 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-2, comment='w2l2', XEweight=2)
exp1.main()

exp2 = SegmentationTrainingApp(epochs=1000, logdir='XE', comment='w2lr3', XEweight=2)
exp2.main()

exp3 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-4, comment='w2lr4', XEweight=2)
exp3.main()

exp4 = SegmentationTrainingApp(epochs=1000, logdir='XE', lr=1e-3, comment='w4lr3', XEweight=4)
exp4.main()

exp5 = SegmentationTrainingApp(epochs=1000, logdir='XE', comment='w6lr3', XEweight=6)
exp5.main()

exp6 = SegmentationTrainingApp(epochs=1000, logdir='XE', comment='w8lr3', XEweight=8)
exp6.main()

exp7 = SegmentationTrainingApp(epochs=2000, logdir='dice', lr=1e-2, comment='dicelr2', loss_fn='dice')
exp7.main()

exp8 = SegmentationTrainingApp(epochs=2000, logdir='dice', lr=1e-3, comment='dicelr3', loss_fn='dice')
exp8.main()

exp9 = SegmentationTrainingApp(epochs=2000, logdir='dice', lr=1e-4, comment='dicelr4', loss_fn='dice')
exp9.main()

# exp9 = SegmentationTrainingApp(epochs=10, logdir='test', lr=1e-4, comment='XEtest', loss_fn='XE', XEweight=2, data_ratio=0.1)
# exp9.main()