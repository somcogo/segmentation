from main import SegmentationTrainingApp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

exp1 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-2, comment='w2lr2', XEweight=2)
exp1.main()

exp2 = SegmentationTrainingApp(epochs=2000, logdir='weights', comment='w2lr3', XEweight=2)
exp2.main()

exp3 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-4, comment='w2lr4', XEweight=2)
exp3.main()

exp4 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-2, comment='w4lr2', XEweight=4)
exp4.main()

exp5 = SegmentationTrainingApp(epochs=2000, logdir='weights', comment='w4lr3', XEweight=4)
exp5.main()

exp6 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-4, comment='w4lr4', XEweight=4)
exp6.main()

exp7 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-2, comment='w8lr2', XEweight=8)
exp7.main()

exp8 = SegmentationTrainingApp(epochs=2000, logdir='weights', comment='w8lr3', XEweight=8)
exp8.main()

exp9 = SegmentationTrainingApp(epochs=2000, logdir='weights', lr=1e-4, comment='w8lr4', XEweight=8)
exp9.main()