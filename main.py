import sys
import argparse
import os
import datetime
import shutil

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import monai
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from models.swinunetr import SwinUNETR
from utils.logconf import logging
from utils.data_loader import getDataLoaderHDF5, getNewDataLoader

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SegmentationTrainingApp:
    def __init__(self, epochs=10, batch_size=16, logdir='test',
                 in_channels=1, lr=1e-3, comment='dwlpt', image_size=64,
                 data_ratio=1.0, XEweight=None, loss_fn='XE'):

        self.epochs = epochs
        self.batch_size = batch_size
        self.logdir_name = logdir
        self.in_channels = in_channels
        self.lr = lr
        self.comment = comment
        self.image_size = image_size
        self.data_ratio = data_ratio
        self.XEweight = XEweight
        self.loss_fn = loss_fn

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)
        if self.XEweight is not None:
            self.XEweight = torch.Tensor([1, self.XEweight])
            self.XEweight = self.XEweight.to(device=self.device)
        else:
            self.XEweight = None
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = SwinUNETR(img_size=self.image_size, patch_size=2, embed_dim=12, depths=[2, 2], num_heads=[3, 6])
        param_num = sum(p.numel() for p in model.parameters())
        log.info('Initiated model with {} params'.format(param_num))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return Adam(params=self.model.parameters(), lr=self.lr)

    def initDl(self):
        return getNewDataLoader(batch_size=self.batch_size)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        train_dl, val_dl = self.initDl()

        val_best = 1e8
        validation_cadence = 5
        for epoch_ndx in range(1, self.epochs + 1):
            logging_index = (epoch_ndx < 10) or (epoch_ndx % 5 == 0 and epoch_ndx < 50) or (epoch_ndx % 50 == 0 and epoch_ndx < 500) or epoch_ndx % 250 == 0

            if logging_index:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics, logging_index=logging_index)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, val_loss, imgs = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics, imgs, logging_index)
                val_best = min(val_loss, val_best)

                self.saveModel('segmentation', epoch_ndx, val_loss == val_best)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(14, len(train_dl.dataset), device=self.device)

        if epoch_ndx < 5:
            log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(train_dl)))

        for batch_ndx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss = self.computeBatchLoss(
                batch_ndx,
                batch_tuple,
                self.batch_size,
                trnMetrics)

            loss.backward()
            self.optimizer.step()

            # if batch_ndx % 10 == 0 and epoch_ndx < 10:
            #     log.info('E{} Training {}/{}'.format(epoch_ndx, batch_ndx, len(train_dl)))

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(14, len(val_dl.dataset), device=self.device)

            if epoch_ndx < 10:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                val_loss, imgs = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    self.batch_size,
                    valMetrics,
                    need_imgs=True
                )
                if batch_ndx % 5 == 0 and epoch_ndx < 10:
                    log.info('E{} Validation {}/{}'.format(epoch_ndx, batch_ndx, len(val_dl)))

        return valMetrics.to('cpu'), val_loss, imgs

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics, need_imgs=False):
        batch, masks = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        masks = masks.to(device=self.device, non_blocking=True)
        masks = masks.long()
        prediction, probabilities = self.model(batch)

        pred_label = torch.argmax(probabilities, dim=1, keepdim=True, out=None)
        pos_pred = pred_label > 0
        neg_pred = ~pos_pred
        pos_mask = masks > 0
        neg_mask = ~pos_mask

        neg_count = neg_mask.sum(dim=[1, 2, 3, 4]).int()
        pos_count = pos_mask.sum(dim=[1, 2, 3, 4]).int()

        true_pos = (pos_pred & pos_mask).sum(dim=[1, 2, 3, 4]).int()
        true_neg = (neg_pred & neg_mask).sum(dim=[1, 2, 3, 4]).int()
        false_pos = neg_count - true_neg
        false_neg = pos_count - true_pos

        dice_score = (2 * true_pos ) / (2 * true_pos + false_pos + false_neg)
        
        precision = torch.zeros(true_pos.shape).to(device=self.device)
        division_mask = true_pos + false_pos > 0
        precision[division_mask] = (true_pos[division_mask] / (true_pos[division_mask] + false_pos[division_mask]))
        assert not ( True in torch.isnan(precision))
        recall = true_pos / (true_pos + false_neg)
        hd_dist = compute_hausdorff_distance(y_pred=pred_label, y=masks, percentile=95, include_background=False)

        if self.loss_fn == 'dice':
            # Trying Dice score as loss function
            loss = 1 - dice_score
            loss.requires_grad = True
        else:
            loss_fn = nn.CrossEntropyLoss(weight=self.XEweight, reduction='none')
            loss = loss_fn(prediction, masks.squeeze(dim=1))

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + masks.size(0)

        if self.loss_fn == 'dice':
            metrics[0, start_ndx:end_ndx] = loss
        else:
            metrics[0, start_ndx:end_ndx] = loss.sum(dim=[1, 2, 3])
            metrics[8, start_ndx:end_ndx] = (loss * pos_mask.squeeze()).sum(dim=[1, 2, 3])
            metrics[9, start_ndx:end_ndx] = (loss * neg_mask.squeeze()).sum(dim=[1, 2, 3])

        metrics[1, start_ndx:end_ndx] = true_pos
        metrics[2, start_ndx:end_ndx] = true_neg
        metrics[3, start_ndx:end_ndx] = false_pos
        metrics[4, start_ndx:end_ndx] = false_neg
        metrics[5, start_ndx:end_ndx] = dice_score
        metrics[6, start_ndx:end_ndx] = precision
        metrics[7, start_ndx:end_ndx] = recall
        metrics[10, start_ndx:end_ndx] = (true_pos + true_neg) / masks.size(-1) ** 3
        metrics[11, start_ndx:end_ndx] = true_pos / pos_count
        metrics[12, start_ndx:end_ndx] = true_neg / neg_count
        metrics[13, start_ndx:end_ndx] = hd_dist.squeeze()

        if need_imgs:
            slice1 = prediction[0, 1, :, :, self.image_size // 2].unsqueeze(0)
            slice2 = prediction[0, 1, :, self.image_size // 2, :].unsqueeze(0)
            slice3 = prediction[0, 1, self.image_size // 2, :, :].unsqueeze(0)
            segmentation1 = pred_label[0, 0, :, :, self.image_size // 2].unsqueeze(0)
            segmentation2 = pred_label[0, 0, :, self.image_size // 2, :].unsqueeze(0)
            segmentation3 = pred_label[0, 0, self.image_size // 2, :, :].unsqueeze(0)
            ground_truth1 = (masks[0, 0, :, :, self.image_size // 2] > 0).unsqueeze(0)
            ground_truth2 = (masks[0, 0, :, self.image_size // 2, :] > 0).unsqueeze(0)
            ground_truth3 = (masks[0, 0, self.image_size // 2, :, :] > 0).unsqueeze(0)
            predicted1 = torch.concat([segmentation1, ground_truth1, ground_truth1], dim=0)
            predicted2 = torch.concat([segmentation2, ground_truth2, ground_truth2], dim=0)
            predicted3 = torch.concat([segmentation3, ground_truth3, ground_truth3], dim=0)
            seg1 = torch.concat([slice1, slice1 * ~ground_truth1, slice1 * ~ground_truth1], dim=0)
            seg2 = torch.concat([slice2, slice2 * ~ground_truth2, slice2 * ~ground_truth2], dim=0)
            seg3 = torch.concat([slice3, slice3 * ~ground_truth3, slice3 * ~ground_truth3], dim=0)
            mask1 = torch.concat([ground_truth1, ground_truth1, ground_truth1], dim=0)
            mask2 = torch.concat([ground_truth2, ground_truth2, ground_truth2], dim=0)
            mask3 = torch.concat([ground_truth3, ground_truth3, ground_truth3], dim=0)
            return loss.mean(), [predicted1, predicted2, predicted3, seg1, seg2, seg3, mask1, mask2, mask3]
        else:
            return loss.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None,
        logging_index=True
    ):
        self.initTensorboardWriters()
        if logging_index:
            log.info("E{} {}:{} loss".format(epoch_ndx, mode_str, metrics[0,:].mean()))

        true_pos = metrics[1,:].sum()
        true_neg = metrics[2,:].sum()
        false_pos = metrics[3,:].sum()
        false_neg = metrics[4,:].sum()
        
        epsilon = 1
        dice_score = (2 * true_pos + epsilon) / (2 * true_pos +  false_pos + false_neg + epsilon)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics[0, :].mean()
        metrics_dict['overall dice'] = dice_score
        metrics_dict['overall precision'] = precision
        metrics_dict['overall recall'] = recall
        metrics_dict['average dice'] = metrics[5, :].mean()
        metrics_dict['average precision'] = metrics[6, :].mean()
        metrics_dict['average recall'] = metrics[7, :].mean()
        metrics_dict['loss/pos'] = metrics[8, :].mean()
        metrics_dict['loss/neg'] = metrics[9, :].mean()
        metrics_dict['correct/all'] = metrics[10, :].mean()
        metrics_dict['correct/pos'] = metrics[11, :].mean()
        metrics_dict['correct/neg'] = metrics[12, :].mean()
        metrics_dict['Hausdorff distance'] = metrics[13, :].mean()

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, global_step=self.totalTrainingSamples_count)

        if img_list:
            grid = torchvision.utils.make_grid(img_list, nrow=3)
            writer.add_image('images',
                grid,
                global_step=self.totalTrainingSamples_count,
                dataformats='CHW')

        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.logdir_name,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.comment,
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

        # log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.logdir_name,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.comment,
                    'best'
                )
            )
            shutil.copyfile(file_path, best_path)

            # log.debug("Saved model params to {}".format(best_path))

if __name__ == '__main__':
    SegmentationTrainingApp().main()
