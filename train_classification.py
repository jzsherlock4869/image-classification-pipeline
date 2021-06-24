import os
import random
import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

from basic_config import config
from utils import cal_acc, get_result, cal_f1score
from utils import seed_everything
from utils import get_adam_optimizer, get_loss_fn, get_lr_scheduler
from utils import ModelSaver, Notebook
from dataset import prepare_dataloader, get_dataset_with_folds

from build_model import BasicClasModel


def train_one_epoch(epoch, model, loss_fn, optimizer, lr_scheduler, train_loader, device, tot_epoch):
    """
    args:
        epoch (int) : num of epoches
        model (nn.Module) : network for training
        loss_fn (nn._Loss) : loss function
        optimizer (nn.optim.Optimizer) : descending optimizer
        lr_scheduler (nn.optim._LRScheduler) : lr policy
        train_loader (DataLoader) : train dataset loader
        device (str) : master device for training
    """
    model.train()
    tot_loss = 0.0
    data_cnt = 0

    tot_step = len(train_loader)
    for step, (imgs, lbls) in enumerate(train_loader):

        imgs = imgs.float()
        lbls = lbls.to(device).long()
        data_cnt += imgs.shape[0]

        outputs = model(imgs)

        loss = loss_fn(outputs, lbls)
        tot_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(train_loader)):
            description = f'train epoch {epoch}/{tot_epoch} | step {step}/{tot_step} | current avg. loss: {tot_loss / data_cnt:.4f} | '
            now = datetime.ctime(datetime.now())
            print('  [TRAIN] >>> ' + description + now )

    lr_scheduler.step()
    avg_loss = tot_loss / data_cnt
    return avg_loss


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):

    model.eval()
    tot_loss = 0.0
    data_cnt = 0
    preds_ls = []
    lbls_ls = []

    tot_step = len(val_loader)
    for step, (imgs, lbls) in enumerate(val_loader):
        imgs = imgs.float()
        lbls = lbls.to(device).long()
        data_cnt += imgs.shape[0]

        outputs = model(imgs)
        loss = loss_fn(outputs, lbls)

        tot_loss += loss

        preds_ls.append(outputs.detach().cpu().numpy())
        lbls_ls.append(lbls.cpu().numpy())

        if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'val epoch {epoch} | step {step}/{tot_step} | current avg. loss: {tot_loss / data_cnt:.4f} | '
            now = datetime.ctime(datetime.now())
            print('    [VALID] >>> ' + description + now)

    y_pred = np.concatenate(preds_ls)
    y_gt = np.concatenate(lbls_ls)
    y_pred = np.argmax(y_pred, 1).astype(np.int)

    test_acc = cal_acc(y_gt, y_pred)
    test_f1 = cal_f1score(y_gt, y_pred)
    test_loss = tot_loss / data_cnt

    print('='*50)
    print('    [VALID FINISH] val loss : {:.4f}, acc : {:.4f}, f1-score : {:.4f}'\
                .format(test_loss, test_acc, test_f1))
    return test_loss, test_acc, test_f1, y_pred


def run_train(config):

    msaver = ModelSaver(config)
    tb_logger = Notebook(config, './tensorboard_logs')
    os.makedirs('./tensorboard_logs', exist_ok=True)

    device_ids = list(range(torch.cuda.device_count()))
    device = torch.device(config.device)
    task_name = config.task_name

    now = datetime.ctime(datetime.now())
    print('Task started | {}'.format(now))

    # select task id here
    train_df, folds = get_dataset_with_folds(task_name)
    oof_df = pd.DataFrame()

    print('Total {} fold training selected ...'.format(config.fold_num))
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):

        if fold_idx not in config.use_folds:
            continue

        valid_folds = train_df.loc[val_idx].reset_index(drop=True)
        print('[FOLD START] Training with label {}, Fold-{} started ...'\
                .format(task_name, fold_idx))
        train_loader, val_loader = prepare_dataloader(train_df, trn_idx, val_idx, config)

        model = BasicClasModel(config).to(device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

        optimizer = get_adam_optimizer(model, config)
        lr_scheduler = get_lr_scheduler(optimizer, config)
        loss_fn = get_loss_fn(device)

        best_loss = 1e10
        max_acc = 0

        for epoch in range(config.epochs):
            train_loss = train_one_epoch(epoch, model, loss_fn, optimizer,\
                                        lr_scheduler, train_loader, device, config.epochs)

            with torch.no_grad():
                val_loss, val_acc, val_f1, preds = valid_one_epoch(epoch, model, loss_fn, val_loader, device)

            if val_loss < best_loss:
                best_loss = val_loss
                print('    [VALID] best val loss occurred: {} in epoch {}'.format(best_loss, epoch))

            info_dict = {'model': model.state_dict(), 'preds': preds}

            # save each epoch
            if max_acc < val_acc:
                max_acc =  val_acc
                print('    [VALID] best acc obtained {:.4f}, F1 score {:.4f}, current epoch {}'\
                            .format(max_acc, val_f1, epoch))
                msaver.save(info_dict, epoch, f'fold_{fold_idx}', is_best=True)
                print('    [VALID] best model saved, current epoch {}'.format(epoch))
            else:
                msaver.save(info_dict, epoch, f'fold_{fold_idx}', is_best=False)

            tb_logger.update({"train_loss": train_loss}, epoch, fold_idx)
            tb_logger.update({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}, epoch, fold_idx)

        preds = msaver.load_best(f'fold_{fold_idx}')['preds']
        valid_folds['preds'] = preds

        print('[FOLD FINISH] ----- label {} fold {} final result ------'.format(task_name, fold_idx))
        fold_tot_acc, fold_tot_f1 = get_result(valid_folds)
        print('[FOLD SUMMARY] Accuracy: {:6f} F1 Score: {:6f}'.format(fold_tot_acc, fold_tot_f1))

        oof_df = pd.concat([oof_df, valid_folds])

        del model, train_loader, val_loader, lr_scheduler, optimizer
        torch.cuda.empty_cache()

    print('----- KFOLD CrossValidation done ------')
    all_fold_acc, all_fold_f1 = get_result(oof_df)
    print('[FINAL RESULT OF EXPERIMENT] Accuracy (all used folds avaraged): {:6f}, F1 Score: {:6f}'\
            .format(all_fold_acc, all_fold_f1))


if __name__ == "__main__":
    seed_everything(config.seed)
    run_train(config)
