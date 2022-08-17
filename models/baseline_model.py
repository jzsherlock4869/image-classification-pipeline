from pkgutil import ImpLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os.path as osp
from copy import deepcopy
import importlib

from archs.timm_arch import TimmArch
from metrics.acc_metric import MetricTopKAcc
from losses.focal_loss import FocalLoss
from utils.net_utils import load_network
from utils.meter_utils import AverageMeter


class BaselineModel:
    """
    Baseline Model:
        input augmented image, output logits
        standard training strategy, from scratch
        loss conducted on logits, focal or celoss
    """
    def __init__(self, opt) -> None:
        self.opt = opt
        self.opt_dataset = opt['datasets']
        self.opt_train = opt['train']
        self.device = opt['device']

        self.num_classes = opt['train']['model_arch']['num_classes']
        self.max_perf = 0.0

    def prepare_training(self):
        os.makedirs(self.opt['log_dir'], exist_ok=True)
        log_path = osp.join(self.opt['log_dir'], self.opt['exp_name'])
        self.writer = SummaryWriter(log_path)
        # prepare dataloader
        self.train_loader, self.val_loader = self.get_trainval_dataloaders(self.opt_dataset)
        # prepare network for training
        opt_model_arch = deepcopy(self.opt_train['model_arch'])
        arch_type = opt_model_arch.pop('type')
        load_path = opt_model_arch.pop('load_path') if 'load_path' in opt_model_arch else None
        self.network = self.get_network(arch_type, load_path, **opt_model_arch).to(self.device)
        if self.opt['multi_gpu']:
            self.network = nn.DataParallel(self.network)
        self.network.train()
        # prepare optimizer and corresponding net params
        opt_optim = deepcopy(self.opt_train['optimizer'])
        optim_type = opt_optim.pop('type')
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print(f'Params {k} will not be optimized.')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **opt_optim)
        # prepare lr scheduler
        opt_scheduler = deepcopy(self.opt_train['scheduler'])
        scheduler_type = opt_scheduler.pop('type')
        self.scheduler = self.get_scheduler(scheduler_type, self.optimizer, **opt_scheduler)
        # prepare criterion
        self.criterion = self.get_criterion(self.opt_train['criterion'])
        # prepare metric for evaluation
        opt_metric = deepcopy(self.opt_train['metric'])
        metric_type = opt_metric.pop('type')
        self.metric = self.get_metric(metric_type)

    def get_trainval_dataloaders(self, opt_dataset):
        opt_train = deepcopy(opt_dataset['train_dataset'])
        opt_val = deepcopy(opt_dataset['val_dataset'])
        train_type = opt_train.pop('type')
        val_type = opt_val.pop('type')
        opt_train['phase'] = 'train'
        opt_val['phase'] = 'val'
        train_loader = getattr(importlib.import_module('data'), train_type)(opt_train)
        val_loader = getattr(importlib.import_module('data'), val_type)(opt_val)
        return train_loader, val_loader

    def get_network(self, arch_type, load_path, **kwargs):
        # get torch original lr_scheduler based on their names
        network = getattr(importlib.import_module('archs'), arch_type)(**kwargs)
        if load_path is not None:
            network = load_network(network, load_path)
            print(f"[MODEL] Load pretrained network from {load_path}")
        else:
            print(f"[MODEL] Network train from scratch")
        return network

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        # get torch original optimizers based on their names
        # e.g. Adam, AdamW, SGD ...
        optimizer = getattr(importlib.import_module('torch.optim'), optim_type)(params, lr, **kwargs)
        return optimizer

    def get_scheduler(self, scheduler_type, optimizer, **kwargs):
        # get torch original lr_scheduler based on their names
        # e.g. CosineAnnealingWarmRestarts, MultiStepLR, ExponentialLR ...
        sch_cls = getattr(importlib.import_module('torch.optim.lr_scheduler'), scheduler_type)
        scheduler = sch_cls(optimizer, **kwargs)
        return scheduler

    def get_criterion(self, opt_criterion):
        crit_type = opt_criterion.pop('type')
        if crit_type.lower() == 'celoss':
            loss_func = nn.CrossEntropyLoss(**opt_criterion)
        elif crit_type.lower() == 'focalloss':
            loss_func = FocalLoss(**opt_criterion)
        else:
            raise NotImplementedError(f'loss func type {crit_type} is currently not supported')
        return loss_func

    def get_metric(self, metric_type, **kwargs):
        if metric_type == 'topk_acc':
            metric = MetricTopKAcc(**kwargs)
        else:
            raise NotImplementedError(f'metric type {metric_type} not implemented yet')
        return metric
    
    # train one epoch
    def train_epoch(self, epoch_id):
        self.network.train()
        batch_size = self.opt_dataset['train_dataset']['batch_size']
        loss_tot_avm = AverageMeter()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # for iter_id, batch in enumerate(self.train_loader):
        for _, batch in pbar:
            # load data mini-batch
            img, label = batch['img'], batch['label']
            img, label = img.to(self.device), label.to(self.device)
            # start optimize
            self.optimizer.zero_grad()
            probs = self.network(img)
            loss = self.criterion(probs, label)
            loss.backward()
            self.optimizer.step()

            loss_tot_avm.update(loss.detach().item(), batch_size)
            # print(f"Train_Loss: {loss_score.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
            pbar.set_postfix(Loss=loss_tot_avm.avg, Epoch=epoch_id, LR=self.optimizer.param_groups[0]['lr'])
            
            self.writer.add_scalar('loss/loss', loss_tot_avm.avg, epoch_id)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch_id)
        
        self.scheduler.step()

    # eval after one epoch
    def eval_epoch(self, epoch_id):
        self.network.eval()
        self.metric.reset()
        print(f"[MODEL] Begin evaluation ...")
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for iter_id, batch in pbar:
                img, label = batch['img'], batch['label']
                img, label = img.to(self.device), label.to(self.device)
                probs = self.network(img)
                self.metric.update(label, probs)
                pbar.set_postfix(Idx=iter_id, NumSamples=self.metric.num_sample())
        # finished all eval images, summarize
        cur_perf = self.metric.calc()
        print("\n\t >>> [MODEL] Evaluate Summary:")
        print(f"  Epoch {epoch_id}, total {len(self.val_loader)} eval images, "
              f"  Metric Type: {self.opt_train['metric']}"
              f"  Eval Score: {cur_perf}")
        self.writer.add_scalar(f'eval/{self.opt_train["metric"]}', cur_perf, epoch_id)
        if cur_perf > self.max_perf:
            print(f"[MODEL] New best performance Epoch {epoch_id} Metric {cur_perf}, save and update")
            self.save_model(epoch_id, cur_perf, copy_best=True)
            self.max_perf = cur_perf

    def save_model(self, epoch_id, val_metric, copy_best=True):
        save_dir = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f'epoch{epoch_id:05}_metric{val_metric:.4f}.pth.tar')
        torch.save(self.network.state_dict(), save_path)
        if copy_best:
            best_path = osp.join(save_dir, 'best.pth.tar')
            torch.save(self.network.state_dict(), best_path)
        return save_path