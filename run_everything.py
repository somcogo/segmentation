from main import SegmentationTrainingApp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 6"

exp2 = SegmentationTrainingApp(epochs=2, logdir='runtest', data_ratio=0.1, comment='firstexp')
exp2.main()

exp1 = SegmentationTrainingApp(epochs=2, batch_size=2, logdir='runtest', lr=2e-4, data_ratio=0.1)
exp1.main()