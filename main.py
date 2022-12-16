import sys
import argparse
import os
import datetime
import shutil

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from models.swinunetr import SwinUNETR
from utils.logconf import logging
from utils.data_loader import getDataLoaderHDF5

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=10, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=1, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')
        parser.add_argument("--image_size", default=64, type=int, help="image size  used for learning")
        parser.add_argument('--data_ratio', default=1.0, type=float, help="what ratio of data to use")

        self.args = parser.parse_args()
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = SwinUNETR(img_size=self.args.image_size, patch_size=2, embed_dim=12, depths=[2, 2], num_heads=[3, 6])
        param_num = sum(p.numel() for p in model.parameters())
        log.info('Initiated model with {} params'.format(param_num))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return Adam(params=self.model.parameters(), lr=self.args.lr)

    def initDl(self):
        return getDataLoaderHDF5(batch_size=self.args.batch_size, image_size=self.args.image_size, num_workers=64, data_ratio=self.args.data_ratio, persistent_workers=True)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        train_dl, val_dl = self.initDl()

        val_best = 1e8
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_dl),
                len(val_dl),
                self.args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, val_loss, imgs = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics, imgs)
                val_best = min(val_loss, val_best)

                self.saveModel('segmentation', epoch_ndx, val_loss == val_best)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(len(train_dl), device=self.device)

        log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(train_dl)))

        for batch_ndx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss = self.computeBatchLoss(
                batch_ndx,
                batch_tuple,
                trnMetrics)

            loss.backward()
            self.optimizer.step()

            if batch_ndx % 10 == 0:
                log.info('E{} Training {}/{}'.format(epoch_ndx, batch_ndx, len(train_dl)))

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(len(val_dl), device=self.device)

            log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                val_loss, imgs = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    valMetrics,
                    need_imgs=True
                )
                if batch_ndx % 5 == 0:
                    log.info('E{} Validation {}/{}'.format(epoch_ndx, batch_ndx, len(val_dl)))

        return valMetrics.to('cpu'), val_loss, imgs

    def computeBatchLoss(self, batch_ndx, batch_tup, metrics, need_imgs=False):
        batch, masks = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        masks = masks.to(device=self.device, non_blocking=True)
        masks = masks.long()
        prediction = self.model(batch)

        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(prediction, masks.squeeze(dim=1))

        metrics[batch_ndx] = loss.detach()

        if need_imgs:
            predicted1 = prediction[0, 0, :, :, self.args.image_size // 2]
            predicted2 = prediction[0, 0, :, self.args.image_size // 2, :]
            predicted3 = prediction[0, 0, self.args.image_size // 2, :, :]
            ground_truth1 = masks[0, 0, :, :, self.args.image_size // 2]
            ground_truth2 = masks[0, 0, :, self.args.image_size // 2, :]
            ground_truth3 = masks[0, 0, self.args.image_size // 2, :, :]
            return loss.mean(), [predicted1, predicted2, predicted3, ground_truth1, ground_truth2, ground_truth3]
        else:
            return loss.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        log.info(
            "E{} {}:{} loss".format(
                epoch_ndx,
                mode_str,
                metrics.mean()
            )
        )

        writer = getattr(self, mode_str + '_writer')
        writer.add_scalar(
            'loss_total',
            scalar_value=metrics.mean(),
            global_step=self.totalTrainingSamples_count
        )

        if img_list:
            img_list = [img.unsqueeze(dim=0).unsqueeze(dim=0) for img in img_list]
            imgs = torch.concat(img_list, dim=0)
            grid = torchvision.utils.make_grid(imgs, nrows=2)
            writer.add_image('images',
                grid,
                global_step=self.totalTrainingSamples_count,
                dataformats='CHW')

        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.args.logdir,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count
        }

        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.args.logdir,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.args.comment,
                    'best'
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

if __name__ == '__main__':
    SegmentationTrainingApp().main()
