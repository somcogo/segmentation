import os
import datetime
import shutil
import copy
import math

import torch
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.logconf import logging
from utils.data_loader import getDataLoaderForOverfitting, getDataLoaderv2
from utils.model_init import model_init
from utils.segmentation_mask import draw_segmenation_mask

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class SegmentationTrainingApp:
    def __init__(self, epochs=10, batch_size=16, logdir='test', lr=1e-3,
                 data_ratio=1., comment='dwlpt', XEweight=None, loss_fn='XE',
                 overfitting=False, model_type='swinunetr',
                 optimizer_type='adam', weight_decay=0, betas=(0.9, 0.999),
                 abs_pos_emb=False, scheduler_type=None, swin_type=1,
                 aug=False, drop_rate=0, attn_drop_rate=0, image_size=64,
                 in_channels=1, T_0=2000, section='large', unet_depth=None):
        
        self.settings = copy.deepcopy(locals())
        del self.settings['self']
        log.info(self.settings)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logdir_name = logdir
        self.data_ratio = data_ratio
        self.comment = comment
        self.loss_fn = loss_fn
        self.overfitting = overfitting
        self.aug = aug
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size, image_size)
        else:
            self.image_size = image_size
        self.section = section

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        if XEweight is not None:
            self.XEweight = torch.tensor([1, XEweight])
            self.XEweight = self.XEweight.to(device=self.device, dtype=torch.float)
        else:
            self.XEweight = None
        self.logdir = os.path.join('./runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model = self.initModel(model_type, swin_type, drop_rate, attn_drop_rate, abs_pos_emb, unet_depth, in_channels)
        self.optimizer = self.initOptimizer(optimizer_type, lr, betas, weight_decay)
        self.scheduler = self.initScheduler(scheduler_type, T_0)

    def initModel(self, model_type, swin_type, drop_rate, attn_drop_rate, ape, unet_depth, in_channels):
        model = model_init(model_type, swin_type, self.image_size, drop_rate, attn_drop_rate, ape, unet_depth, in_channels)
        param_num = sum(p.numel() for p in model.parameters())
        log.info('Initiated model with {} params'.format(param_num))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self, optimizer_type, lr, betas, weight_decay):
        if optimizer_type == 'adam':
            return Adam(params=self.model.parameters(), lr=lr, betas=betas)
        elif optimizer_type == 'adamw':
            return AdamW(params=self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def initScheduler(self, scheduler_type, T_0):
        if scheduler_type == 'cosinewarmre':
            return CosineAnnealingWarmRestarts(self.optimizer, T_0)
        else:
            return None

    def initDl(self):
        if self.overfitting:
            return getDataLoaderForOverfitting(batch_size=10, image_size=self.image_size, num_workers=64, persistent_workers=False, aug=self.aug)
        else:
            # return getDataLoaderHDF5(batch_size=self.batch_size, image_size=self.image_size, num_workers=64, data_ratio=self.data_ratio, persistent_workers=True, aug=self.aug)
            # return getNewDataLoader(batch_size=self.batch_size, persistent_workers=True, aug=self.aug)
            return getDataLoaderv2(batch_size=self.batch_size, persistent_workers=True, aug=self.aug, section=self.section)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.comment)

    def main(self):
        log.info("Starting {}".format(type(self).__name__))

        train_dl, val_dl = self.initDl()

        val_best = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.epochs + 1):
            logging_index = epoch_ndx % 10**(math.floor(math.log(epoch_ndx, 10))) == 0

            trnMetrics = self.doTraining(train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, val_loss, dice_score, imgs = self.doValidation(val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics, imgs)
                val_best = max(dice_score, val_best)

                self.saveModel('segmentation', epoch_ndx, valMetrics, dice_score == val_best)

                if logging_index:
                    log.info("Epoch {} of {}, val loss {}, dice score {}".format(
                        epoch_ndx,
                        self.epochs,
                        val_loss,
                        dice_score,
                    ))
            
            if self.scheduler is not None:
                self.scheduler.step()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(14, len(train_dl.dataset), device=self.device)

        for batch_ndx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()
            loss = self.computeBatchLoss(
                batch_ndx,
                batch_tuple,
                self.batch_size,
                trnMetrics)

            loss.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics.to('cpu')

    def doValidation(self, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(14, len(val_dl.dataset), device=self.device)

            for batch_ndx, batch_tuple in enumerate(val_dl):
                val_loss, dice_score, imgs = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    self.batch_size,
                    valMetrics,
                    need_imgs=True
                )

        return valMetrics.to('cpu'), val_loss, dice_score, imgs
    
    def diceLoss(self, prediction_g, label_g, epsilon=1e-5):
        diceLabel_g = label_g.sum(dim=[1,2,3,4])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3,4])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3,4])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g
    
    def computeMetrics(self, pred_label, mask):
        eps = 1e-5

        true_pos = (mask * pred_label).sum(dim=[1, 2, 3, 4]).unsqueeze(0)
        false_pos = ((1 - mask)*pred_label).sum(dim=[1, 2, 3, 4]).unsqueeze(0)
        false_neg = (mask*(1 - pred_label)).sum(dim=[1, 2, 3, 4]).unsqueeze(0)

        dice_score = (2*true_pos + eps)/(2*true_pos + false_pos + false_neg + eps)
        precision = (true_pos + eps)/(true_pos + false_pos + eps)
        recall = (true_pos + eps)/(true_pos + false_neg + eps)

        correct = (mask == pred_label).int().sum(dim=[1, 2, 3, 4])
        correct_pos = ((mask == pred_label)*mask).sum(dim=[1, 2, 3, 4])
        correct_neg = ((mask == pred_label)*(1 - mask)).sum(dim=[1, 2, 3, 4])
        pos_count = mask.sum(dim=[1, 2, 3, 4])
        neg_count = (1 - mask).sum(dim=[1, 2, 3, 4])
        correct_per_all = ((correct + eps)/(pos_count + neg_count + eps)).unsqueeze(0)
        correct_per_pos = ((correct_pos + eps)/(pos_count + eps)).unsqueeze(0)
        correct_per_neg = ((correct_neg + eps)/(neg_count + eps)).unsqueeze(0)

        metrics = torch.concat([true_pos, false_pos, false_neg, dice_score, precision, recall, correct_per_all, correct_per_pos, correct_per_neg], dim=0)
        return metrics

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics, need_imgs=False):
        batch, masks, img_ids = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        masks = masks.to(device=self.device, non_blocking=True).long()
        prediction, probabilities = self.model(batch)

        pred_label = torch.argmax(probabilities, dim=1, keepdim=True, out=None)
        loss_fn = CrossEntropyLoss(weight=self.XEweight, reduction='none')
        loss = loss_fn(prediction, masks.squeeze(dim=1))

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + masks.size(0)
        pos_mask = masks > 0
        neg_mask = ~pos_mask
        metrics[0, start_ndx:end_ndx] = loss.sum(dim=[1, 2, 3])
        metrics[1, start_ndx:end_ndx] = (loss * pos_mask.squeeze()).sum(dim=[1, 2, 3])
        metrics[2, start_ndx:end_ndx] = (loss * neg_mask.squeeze()).sum(dim=[1, 2, 3])
        metrics[3:12, start_ndx:end_ndx] = self.computeMetrics(pred_label, masks)

        if need_imgs:
            combined_masks = torch.concat([masks[0], pred_label[0]], dim=0)
            rgb_img = torch.concat([batch[0], batch[0], batch[0]], dim=0)
            rgb_img = 255*(rgb_img - rgb_img.min())/(rgb_img.max()- rgb_img.min())
            segmentation_mask = draw_segmenation_mask(rgb_img, combined_masks, torch.tensor([[255, 0, 0], [0, 255, 0]], device=self.device), device=self.device)

            x, y, z = torch.where(masks[0])[-3:] if masks[0].sum() > 0 else torch.tensor(self.image_size)//2
            x, y, z, = x.float().mean().int(), y.float().mean().int(), z.float().mean().int()
            im1 = segmentation_mask[:, :, :, z]
            im2 = segmentation_mask[:, :, y, :]
            im3 = segmentation_mask[:, x, :, :]

            # slice1 = prediction[0, 1, :, :, self.image_size[2] // 2].unsqueeze(0)
            # slice2 = prediction[0, 1, :, self.image_size[1] // 2, :].unsqueeze(0)
            # slice3 = prediction[0, 1, self.image_size[0] // 2, :, :].unsqueeze(0)
            # segmentation1 = pred_label[0, 0, :, :, self.image_size[2] // 2].unsqueeze(0)
            # segmentation2 = pred_label[0, 0, :, self.image_size[1] // 2, :].unsqueeze(0)
            # segmentation3 = pred_label[0, 0, self.image_size[0] // 2, :, :].unsqueeze(0)
            # ground_truth1 = (masks[0, 0, :, :, self.image_size[2] // 2] > 0).unsqueeze(0)
            # ground_truth2 = (masks[0, 0, :, self.image_size[1] // 2, :] > 0).unsqueeze(0)
            # ground_truth3 = (masks[0, 0, self.image_size[0] // 2, :, :] > 0).unsqueeze(0)
            # image1 = batch[0, 0, :, :, self.image_size[2] // 2].unsqueeze(0)
            # image2 = batch[0, 0, :, self.image_size[1] // 2, :].unsqueeze(0)
            # image3 = batch[0, 0, self.image_size[0] // 2, :, :].unsqueeze(0)
            # predicted1 = torch.concat([ground_truth1, segmentation1, segmentation1], dim=0)
            # predicted2 = torch.concat([ground_truth2, segmentation2, segmentation2], dim=0)
            # predicted3 = torch.concat([ground_truth3, segmentation3, segmentation3], dim=0)
            # seg1 = torch.concat([slice1, slice1 * ~ground_truth1, slice1 * ~ground_truth1], dim=0)
            # seg2 = torch.concat([slice2, slice2 * ~ground_truth2, slice2 * ~ground_truth2], dim=0)
            # seg3 = torch.concat([slice3, slice3 * ~ground_truth3, slice3 * ~ground_truth3], dim=0)
            # mask1 = torch.concat([ground_truth1, ground_truth1, ground_truth1], dim=0)
            # mask2 = torch.concat([ground_truth2, ground_truth2, ground_truth2], dim=0)
            # mask3 = torch.concat([ground_truth3, ground_truth3, ground_truth3], dim=0)
            # ct1 = torch.concat([image1, image1, image1], dim=0)
            # ct2 = torch.concat([image2, image2, image2], dim=0)
            # ct3 = torch.concat([image3, image3, image3], dim=0)
            # return loss.mean(), metrics[6].mean(), [predicted1, seg1, mask1, predicted2, seg2, mask2, predicted3, seg3, mask3]
            return loss.mean(), metrics[6].mean(), [im1, im2, im3]
        else:
            return loss.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None,
    ):
        self.initTensorboardWriters()

        true_pos = metrics[3,:].sum()
        false_pos = metrics[4,:].sum()
        false_neg = metrics[5,:].sum()
        
        epsilon = 1e-5
        dice_score = (2 * true_pos + epsilon) / (2 * true_pos +  false_pos + false_neg + epsilon)
        precision = (true_pos + epsilon) / (true_pos + false_pos + epsilon)
        recall = (true_pos + epsilon) / (true_pos + false_neg + epsilon)

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics[0, :].mean()
        metrics_dict['loss/pos'] = metrics[1, :].mean()
        metrics_dict['loss/neg'] = metrics[2, :].mean()
        metrics_dict['overall/dice'] = dice_score
        metrics_dict['overall/precision'] = precision
        metrics_dict['overall/recall'] = recall
        metrics_dict['average/dice'] = metrics[6, :].mean()
        metrics_dict['average/precision'] = metrics[7, :].mean()
        metrics_dict['average/recall'] = metrics[8, :].mean()
        metrics_dict['correct/all'] = metrics[9, :].mean()
        metrics_dict['correct/pos'] = metrics[10, :].mean()
        metrics_dict['correct/neg'] = metrics[11, :].mean()
        metrics_dict['Hausdorff distance'] = metrics[13, :].mean()

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, global_step=epoch_ndx)
        if img_list:
            for ndx in range(3):
                # grid = make_grid(img_list[3*ndx: 3*(ndx+1)], nrow=3)
                writer.add_image('images/{}'.format(ndx),
                    img_list[ndx],
                    global_step=epoch_ndx,
                    dataformats='CHW')

        writer.flush()

    def saveModel(self, type_str, epoch_ndx, val_metrics, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.logdir_name,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.comment,
                epoch_ndx
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
            'valmetrics':val_metrics
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
