import os
import datetime
import shutil
import copy
import math

import numpy as np
import torch
from torch.utils.data import DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from utils.logconf import logging
from utils.data_loader import getSegmentationDataLoader, getSwarmSegmentationDataLoader
from utils.model_init import model_init
from utils.segmentation_mask import draw_segmenation_mask
from utils.scheduler import get_cosine_lr_with_linear_warmup
from utils.merge_strategies import get_layer_list
from inference import do_inference_save_results

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class SegmentationTrainingApp:
    def __init__(self, epochs=10, batch_size=16, logdir='test', lr=1e-3,
                 data_ratio=1., comment='dwlpt', XEweight=None, loss_fn='XE',
                 model_type='swinunetr', optimizer_type='adam', weight_decay=0,
                 betas=(0.9, 0.999), abs_pos_emb=False, scheduler_type=None,
                 swin_type=1, aug='nnunet', drop_rate=0, attn_drop_rate=0,
                 image_size=64, in_channels=1, T_0=2000, section='large',
                 unet_depth=None, pretrained=True, swarm_training=False,
                 model_path=None, grad_accumulation=1, foreground_pref_chance=0.,
                 swarm_strat='all', site_repetition=None, iterations=None,
                 dataset='111', distr=None):
        
        # self.settings = copy.deepcopy(locals())
        # del self.settings['self']
        # log.info(self.settings)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logdir_name = logdir
        self.data_ratio = data_ratio
        self.comment = comment
        self.loss_fn = loss_fn
        self.model_type = model_type
        assert aug in ['nnunet', 'old']
        self.aug = aug
        self.image_size = [image_size, image_size, image_size] if isinstance(image_size, int) else  image_size
        self.section = section
        self.swarm_training = swarm_training
        self.grad_accumulation = grad_accumulation
        self.foreground_pred_chance = foreground_pref_chance
        self.T_0 = T_0 if T_0 is not None else epochs
        self.model_path = model_path
        self.strategy = swarm_strat
        self.site_repetition = site_repetition
        self.iterations = [iterations, iterations, iterations] if isinstance(iterations, int) else iterations
        self.dataset = dataset
        self.distr = distr
        self.main_process = distr is None or distr['rank'] == 0

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cpu' if not self.use_cuda else 'cuda' if self.distr is None else distr['rank']
        self.XEweight = torch.tensor([1, XEweight]).to(device=self.device, dtype=torch.float) if XEweight is not None else None
        self.logdir = os.path.join('/home/hansel/developer/segmentation/runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)

        if self.main_process:
            self.trn_writer = SummaryWriter(
                log_dir=os.path.join('/home/hansel/developer/segmentation/runs', logdir, 'trn-' + comment)
            )
            self.val_writer = SummaryWriter(
                log_dir=os.path.join('/home/hansel/developer/segmentation/runs', logdir, 'val-' + comment)
            )
        self.totalTrainingSamples_count = 0

        if swarm_training:
            self.models = self.initModel(model_type, swin_type, drop_rate, attn_drop_rate, abs_pos_emb, unet_depth, in_channels, pretrained)
            self.optimizers = self.initOptimizer(optimizer_type, lr, betas, weight_decay)
            self.schedulers = self.initScheduler(scheduler_type, T_0)
            self.mergeModels(is_init=True, model_path=model_path)
        else:
            self.model = self.initModel(model_type, swin_type, drop_rate, attn_drop_rate, abs_pos_emb, unet_depth, in_channels, pretrained)
            self.optimizer = self.initOptimizer(optimizer_type, lr, betas, weight_decay)
            self.scheduler = self.initScheduler(scheduler_type, T_0)

    def initModel(self, model_type, swin_type, drop_rate, attn_drop_rate, ape, unet_depth, in_channels, pretrained):
        if self.swarm_training:
            models = [model_init(model_type, swin_type, self.image_size, drop_rate, attn_drop_rate, ape, unet_depth, in_channels, pretrained) for _ in range(3)]
        else:
            model = model_init(model_type, swin_type, self.image_size, drop_rate, attn_drop_rate, ape, unet_depth, in_channels, pretrained)
            models = None
        if self.model_path is not None:
            state_dict = torch.load(self.model_path)['model_state']
            model.load_state_dict(state_dict)
        if self.use_cuda and self.distr is None:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if self.swarm_training:
                for i, model in enumerate(models):
                    if torch.cuda.device_count() > 1:
                        model = DataParallel(model)
                    model = model.to(self.device)
                    models[i] = model
            else:
                if torch.cuda.device_count() > 1:
                    model = DataParallel(model)
                model = model.to(self.device)
        elif self.use_cuda and self.distr is not None:
            if self.main_process:
                log.info(f"Using CUDA; distributed learning on {self.distr['world_size']} devices.")
            if self.swarm_training:
                for i, model in enumerate(models):
                    if torch.cuda.device_count() > 1:
                        model = DistributedDataParallel(model.to(self.device), device_ids=[self.distr['rank']], broadcast_buffers=False)
                        # model = torch.compile(model)
                    models[i] = model
            else:
                if torch.cuda.device_count() > 1:
                    model = DistributedDataParallel(model.to(self.device), device_ids=[self.distr['rank']], broadcast_buffers=False, find_unused_parameters=True)
                model = model.to(self.device)
        param_num = sum(p.numel() for p in model.parameters())
        if self.main_process:
            log.info('Initiated model with {} params'.format(param_num))
        out = models if models is not None else model
        return out

    def initOptimizer(self, optimizer_type, lr, betas, weight_decay):
        opt_class = Adam if optimizer_type == 'adam' else AdamW
        if self.swarm_training:
            lr = lr if isinstance(lr, list) else [lr, lr, lr]
            return [opt_class(params=model.parameters(), lr=lr[ndx], betas=betas, weight_decay=weight_decay) for ndx, model in enumerate(self.models)]
        else:
            return opt_class(params=self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def initScheduler(self, scheduler_type, T_0):
        if self.swarm_training:
            if scheduler_type == 'cosinewarmre':
                return [CosineAnnealingWarmRestarts(opt, T_0) for opt in self.optimizers]
            elif scheduler_type == 'warmupcosine':
                return [get_cosine_lr_with_linear_warmup(optim=opt, warm_up_epochs=20, T_max=self.T_0-20, eta_min=1e-6) for opt in self.optimizers]
        else:
            if scheduler_type == 'cosinewarmre':
                return CosineAnnealingWarmRestarts(self.optimizer, T_0)
            elif scheduler_type == 'warmupcosine':
                return get_cosine_lr_with_linear_warmup(optim=self.optimizer, warm_up_epochs=20, T_max=self.T_0-20, eta_min=1e-6)
        return None

    def initDl(self):
        if self.swarm_training:
            return getSwarmSegmentationDataLoader(batch_size=self.batch_size, aug=self.aug, section=self.section, image_size=self.image_size, foreground_pref_chance=self.foreground_pred_chance, dataset=self.dataset, distr=self.distr)
        else:
            return getSegmentationDataLoader(batch_size=self.batch_size, aug=self.aug, section=self.section, image_size=self.image_size, foreground_pref_chance=self.foreground_pred_chance, dataset=self.dataset, distr=self.distr)

    def main(self):
        if self.main_process:
            log.info("Starting {}".format(type(self).__name__))

        trn_dl, val_dl, trn_sampler, val_sampler = self.initDl()

        val_best = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.epochs + 1):
            logging_index = epoch_ndx % 10**(math.floor(math.log(epoch_ndx, 10))) == 0

            if self.distr is not None and self.swarm_training:
                for t_sampler, v_sampler in zip(trn_sampler, val_sampler):
                    t_sampler.set_epoch(epoch_ndx)
                    v_sampler.set_epoch(epoch_ndx)
            elif self.distr is not None:
                trn_sampler.set_epoch(epoch_ndx)
                val_sampler.set_epoch(epoch_ndx)

            if self.swarm_training:
                trnMetrics = self.doSwarmTraining(trn_dl)
                self.mergeModels()
            else:
                trnMetrics = self.doTraining(trn_dl)

            if self.main_process:
                dist.reduce(trnMetrics, op=dist.ReduceOp.SUM, dst=0)
                trnMetrics = trnMetrics.to('cpu') if self.distr is None else trnMetrics.to('cpu') / self.distr['world_size']
                self.logMetrics(epoch_ndx, 'trn', trnMetrics)
            
            if epoch_ndx < self.T_0:
                if self.swarm_training:
                    if self.schedulers is not None:
                        for scheduler in self.schedulers:
                            scheduler.step()
                else:
                    if self.scheduler is not None:
                        self.scheduler.step()

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                if self.swarm_training:
                    valMetrics, val_loss, imgs = self.doSwarmValidation(val_dl)
                else:
                    valMetrics, val_loss, imgs = self.doValidation(val_dl)

                if self.main_process:
                    dist.reduce(valMetrics, op=dist.ReduceOp.SUM, dst=0)
                    valMetrics = valMetrics.to('cpu') if self.distr is None else trnMetrics.to('cpu') / self.distr['world_size']
                    val_dice = self.logMetrics(epoch_ndx, 'val', valMetrics, imgs)
                    val_best = max(val_dice, val_best)

                    self.saveModel('segmentation', epoch_ndx, valMetrics, val_dice == val_best)

                    if logging_index:
                        log.info("Epoch {} of {}, val loss {}, dice score {}".format(
                            epoch_ndx,
                            self.epochs,
                            val_loss,
                            val_dice,
                        ))
            if self.distr is not None:
                dist.barrier()

        if self.epochs > 100 and self.main_process:
            log.info('Training finished. Performing inference on validation set')
            save_path = os.path.join(
                'saved_metrics',
                'inference',
                self.logdir_name,
                'segmentation_{}_{}-gaussian-inference'.format(
                    self.time_str,
                    self.comment,
                )
            )
            os.makedirs(os.path.dirname(save_path), mode=0o755, exist_ok=True)
            do_inference_save_results(save_path=save_path, model=self.models[0] if self.swarm_training else self.model, image_size=self.image_size, log=False, gaussian_weights=True)

        if self.main_process:
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, train_dl):
        self.model.train()
        trnMetrics = torch.zeros(13, len(train_dl.dataset), device=self.device)

        for batch_ndx, batch_tuple in enumerate(train_dl):
            loss, _ = self.computeBatchLoss(
                self.model,
                batch_ndx,
                batch_tuple,
                self.batch_size,
                trnMetrics)

            loss.backward()
            if (batch_ndx + 1) % self.grad_accumulation == 0 or (batch_ndx + 1) == len(train_dl):
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics

    def doSwarmTraining(self, train_dl):
        local_metrics = [torch.zeros(13, len(dl.dataset), device=self.device) for dl in train_dl]

        for site, dl in enumerate(train_dl):
            model = self.models[site]
            optim = self.optimizers[site]
            model.train()
            if self.iterations is not None:
                iters = 0
                while iters < self.iterations[site]:
                    for batch_ndx, batch_tuple in enumerate(dl):
                        loss, _ = self.computeBatchLoss(
                            model,
                            batch_ndx,
                            batch_tuple,
                            self.batch_size,
                            local_metrics[site])

                        loss.backward()
                        if (batch_ndx + 1) % self.grad_accumulation == 0 or (batch_ndx + 1) == len(train_dl):
                            optim.step()
                            optim.zero_grad()
                        iters += 1
                        if iters >= self.iterations[site]:
                            break
            elif self.site_repetition is not None:
                for _ in range(self.site_repetition[site]):
                    for batch_ndx, batch_tuple in enumerate(dl):
                        loss, _ = self.computeBatchLoss(
                            model,
                            batch_ndx,
                            batch_tuple,
                            self.batch_size,
                            local_metrics[site])

                        loss.backward()
                        if (batch_ndx + 1) % self.grad_accumulation == 0 or (batch_ndx + 1) == len(train_dl):
                            optim.step()
                            optim.zero_grad()


        trn_metrics = torch.concat(local_metrics, dim=1)
        self.totalTrainingSamples_count += trn_metrics.shape[1]

        return trn_metrics

    def doValidation(self, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(13, len(val_dl.dataset), device=self.device)

            for batch_ndx, batch_tuple in enumerate(val_dl):
                if batch_ndx == 1:
                    need_imgs = True
                else:
                    need_imgs = False
                val_loss, imgs = self.computeBatchLoss(
                    self.model,
                    batch_ndx,
                    batch_tuple,
                    self.batch_size,
                    valMetrics,
                    need_imgs=need_imgs
                )
                if imgs is not None:
                    imgs_to_save = imgs

        return valMetrics, val_loss, imgs_to_save

    def doSwarmValidation(self, val_dl):
        local_metrics = [torch.zeros(13, len(dl.dataset), device=self.device) for dl in val_dl]
        # valMetrics = torch.zeros(14, len(val_dl.dataset), device=self.device)
        for site, dl in enumerate(val_dl):
            with torch.no_grad():
                self.models[site].eval()

                for batch_ndx, batch_tuple in enumerate(dl):
                    if batch_ndx == 1 and site == 0:
                        need_imgs = True
                    else:
                        need_imgs = False
                    val_loss, imgs = self.computeBatchLoss(
                        self.models[site],
                        batch_ndx,
                        batch_tuple,
                        self.batch_size,
                        local_metrics[site],
                        need_imgs=need_imgs
                    )
                    if imgs is not None:
                        imgs_to_save = imgs
        
        val_metrics = torch.concat(local_metrics, dim=1)

        return val_metrics, val_loss, imgs_to_save
    
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

    def computeBatchLoss(self, model, batch_ndx, batch_tup, batch_size, metrics, need_imgs=False):
        batch, masks = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        masks = masks.to(device=self.device, non_blocking=True).long()
        if 'mednext' in self.model_type or 'monai' in self.model_type:
            prediction = model(batch)
        else:
            prediction, probabilities = model(batch)

        pred_label = torch.argmax(prediction, dim=1, keepdim=True, out=None)
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
            combined_masks = torch.concat([masks[0], pred_label[0]], dim=0).detach().cpu().numpy()
            img_cpu = batch[0].squeeze().detach().cpu().numpy()
            segmentation_mask = draw_segmenation_mask(img_cpu, combined_masks, np.array([[255, 0, 0], [0, 255, 0]]), alpha=0.3)

            x, y, z = torch.where(masks[0])[-3:] if masks[0].sum() > 0 else torch.tensor(self.image_size)//2
            x, y, z, = x.float().mean().int(), y.float().mean().int(), z.float().mean().int()
            im1 = segmentation_mask[:, :, :, z]
            im2 = segmentation_mask[:, :, y, :]
            im3 = segmentation_mask[:, x, :, :]
            return loss.sum(), [im1, im2, im3]
        else:
            return loss.sum(), None

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None,
    ):
        metrics_dict = self.convert_metrics_to_dict(metrics, 'all')
        if self.swarm_training:
            alt_end = 138 if mode_str == 'trn' else 34
            neu_end = 252 if mode_str == 'trn' else 63
            metrics_dict.update(self.convert_metrics_to_dict(metrics[:, :alt_end], tag='alt'))
            metrics_dict.update(self.convert_metrics_to_dict(metrics[:, alt_end:neu_end], tag='neu'))
            metrics_dict.update(self.convert_metrics_to_dict(metrics[:, neu_end:], tag='asbach'))

        dice_score = metrics_dict['overall/dice-all']

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, global_step=epoch_ndx)
        if img_list is not None:
            for ndx in range(3):
                writer.add_image('images/{}'.format(ndx),
                    img_list[ndx],
                    global_step=epoch_ndx,
                    dataformats='CHW')

        writer.flush()
        return dice_score
    
    def convert_metrics_to_dict(self, metrics, tag):
        true_pos = metrics[3,:].sum()
        false_pos = metrics[4,:].sum()
        false_neg = metrics[5,:].sum()
        
        epsilon = 1e-5
        dice_score = (2 * true_pos + epsilon) / (2 * true_pos +  false_pos + false_neg + epsilon)
        precision = (true_pos + epsilon) / (true_pos + false_pos + epsilon)
        recall = (true_pos + epsilon) / (true_pos + false_neg + epsilon)

        metrics_dict = {}
        metrics_dict['loss/all-{}'.format(tag)] = metrics[0, :].mean()
        metrics_dict['loss/pos-{}'.format(tag)] = metrics[1, :].mean()
        metrics_dict['loss/neg-{}'.format(tag)] = metrics[2, :].mean()
        metrics_dict['overall/dice-{}'.format(tag)] = dice_score
        metrics_dict['overall/precision-{}'.format(tag)] = precision
        metrics_dict['overall/recall-{}'.format(tag)] = recall
        metrics_dict['average/dice-{}'.format(tag)] = metrics[6, :].mean()
        metrics_dict['average/precision-{}'.format(tag)] = metrics[7, :].mean()
        metrics_dict['average/recall-{}'.format(tag)] = metrics[8, :].mean()
        metrics_dict['correct/all-{}'.format(tag)] = metrics[9, :].mean()
        metrics_dict['correct/pos-{}'.format(tag)] = metrics[10, :].mean()
        metrics_dict['correct/neg-{}'.format(tag)] = metrics[11, :].mean()

        return metrics_dict

    def mergeModels(self, is_init=False, model_path=None):
        if is_init:
            if model_path is not None:
                loaded_dict = torch.load(model_path)
                if 'model_state' in loaded_dict.keys():
                    state_dict = loaded_dict['model_state']
                else:
                    state_dict = loaded_dict[0]['model_state']
            else:
                state_dict = self.models[0].state_dict()
            if 'embedding.weight' in '\t'.join(state_dict.keys()):
                state_dict['embedding.weight'] = state_dict['embedding.weight'][0].unsqueeze(0).repeat(3, 1)
            for model in self.models:
                model.load_state_dict(state_dict, strict=False)
        else:
            original_list = [name for name, _ in self.models[0].named_parameters()]
            layer_list = get_layer_list(model=self.model_type, strategy=self.strategy, original_list=original_list)
            state_dicts = [model.state_dict() for model in self.models]
            param_dict = {layer: torch.zeros(state_dicts[0][layer].shape, device=self.device) for layer in layer_list}

            for layer in layer_list:
                for state_dict in state_dicts:
                    param_dict[layer] += state_dict[layer]
                param_dict[layer] /= len(state_dicts)

            for model in self.models:
                model.load_state_dict(param_dict, strict=False)

    def saveModel(self, type_str, epoch_ndx, val_metrics, isBest=False):
        if isBest:
            model_file_path = os.path.join(
                '/home/hansel/developer/segmentation/saved_models',
                self.logdir_name,
                '{}_{}_{}.state'.format(
                    type_str,
                    self.time_str,
                    self.comment
                )
            )
            os.makedirs(os.path.dirname(model_file_path), mode=0o755, exist_ok=True)
            data_file_path = os.path.join(
                '/home/hansel/developer/segmentation/saved_metrics',
                self.logdir_name,
                '{}_{}_{}.pt'.format(
                    type_str,
                    self.time_str,
                    self.comment
                )
            )
            os.makedirs(os.path.dirname(data_file_path), mode=0o755, exist_ok=True)

            if self.swarm_training:
                model = self.models[0]
            else:
                model = self.model
            if isinstance(model, DataParallel):
                model = model.module

            model_state = {
                'model_state': model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state': self.optimizers[0].state_dict() if self.swarm_training else self.optimizer.state_dict(),
                'optimizer_name': type(self.optimizers[0]).__name__ if self.swarm_training else type(self.optimizer).__name__,
                'scheduler_state': self.schedulers[0].state_dict() if self.swarm_training else self.scheduler.state_dict(),
                'scheduler_name': type(self.schedulers[0]).__name__ if self.swarm_training else type(self.scheduler).__name__,
                'epoch': epoch_ndx,
            }
            data_state = {
                'epoch': epoch_ndx,
                'totalTrainingSamples_count': self.totalTrainingSamples_count,
                'valmetrics':val_metrics.detach().cpu()
            }

            torch.save(model_state, model_file_path)
            torch.save(data_state, data_file_path)

            # log.debug("Saved model params to {}".format(file_path))

if __name__ == '__main__':
    SegmentationTrainingApp().main()
