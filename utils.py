import random
import torch
import os
import numpy as np
import torch.nn as nn
from basic_config import config
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def cal_acc(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_pred)
    return acc

def cal_f1score(y_true, y_pred):
    # micro f1 is like pIoU, macro like mIoU
    return f1_score(y_true, y_pred, average='macro')

def get_result(res_df):
    y_pred=res_df['preds'].values
    y_true=res_df[config.target_col].values
    return cal_acc(y_true, y_pred), cal_f1score(y_true, y_pred)

def get_loss_fn(device):
    return nn.CrossEntropyLoss().to(device)

def get_lr_scheduler(optimizer, config):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.T_0, 
        T_mult=1,
        eta_min=config.min_lr, 
        last_epoch=-1)
    return scheduler

def get_adam_optimizer(model, config):
    optimizer = Adam(model.parameters(),
                        lr=config.lr, 
                        weight_decay=config.weight_decay,
                        amsgrad=False)
    return optimizer

class ModelSaver:
    """
    train model saver
    """
    def __init__(self, config):

        self.save_path = config.OUTPUT_DIR
        self.model_arch = config.model_arch
        EXPNAME_ORI = config.EXP_NAME

        os.makedirs(self.save_path, exist_ok=True)
        subdir_names = os.listdir(self.save_path)
        count = 0
        EXPNAME = EXPNAME_ORI
        while EXPNAME in subdir_names:
            count += 1
            EXPNAME = EXPNAME_ORI + "_" + str(count)
        os.makedirs(os.path.join(self.save_path, EXPNAME), exist_ok=False)

        self.EXPNAME = EXPNAME
    
    def save(self, info_dict, step, postfix, is_best=False):
        save_filename = os.path.join(self.save_path, self.EXPNAME,\
             'arch_{}_step{}_ps_{}.ckpt'.format(self.model_arch, step, postfix))
        torch.save(info_dict, save_filename)
        if is_best:
            best_save_filename = os.path.join(self.save_path, self.EXPNAME,\
                 'arch_{}_best_ps_{}.ckpt'.format(self.model_arch, postfix))
            torch.save(info_dict, best_save_filename)

    def load_best(self, postfix):
        best_save_filename = os.path.join(self.save_path, self.EXPNAME,\
                 'arch_{}_best_ps_{}.ckpt'.format(self.model_arch, postfix))
        best_dict = torch.load(best_save_filename)
        return best_dict


class Notebook:
    """
    tensorbaord logger
    """
    def __init__(self, config, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        EXPNAME_ORI = config.EXP_NAME + '_' + config.model_arch
        subdir_names = os.listdir(log_dir)
        count = 0
        EXPNAME = EXPNAME_ORI
        while EXPNAME in subdir_names:
            count += 1
            EXPNAME = EXPNAME_ORI + "_" + str(count)
        os.makedirs(os.path.join(log_dir, EXPNAME), exist_ok=False)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, EXPNAME))
    
    def update(self, info_dict, step, fold):
        if "train_loss" in info_dict:
            self.writer.add_scalar("loss/train/fold_{}".format(fold), info_dict["train_loss"], step)
        else:
            assert "val_loss" in info_dict
            self.writer.add_scalar("loss/eval/fold_{}".format(fold), info_dict["val_loss"], step)
            self.writer.add_scalar("acc/eval/fold_{}".format(fold), info_dict["val_acc"], step)
            self.writer.add_scalar("f1score/eval/fold_{}".format(fold), info_dict["val_f1"], step)

