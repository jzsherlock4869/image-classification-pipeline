import sys
sys.path.append('../')
sys.path.append('./')

import os
import os.path as osp
import random
from glob import glob
import argparse
import yaml, addict
import numpy as np
from datetime import datetime
import importlib

import torch
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings('ignore')

# set used config .yml file path
parser = argparse.ArgumentParser(description='input the configure file path')
parser.add_argument('-opt', type=str, required=True, help='config file path')
args = parser.parse_args()
config_path = args.opt

# load configs
with open(config_path, 'r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)
opt = addict.Dict(opt)
# modify params in config on the fly, set exp_name to config filename
opt['exp_name'] = os.path.basename(config_path)[:-4]

ckpt_path = opt['model_arch']['load_path']
# make dirs for save
result_dir = osp.join(opt['result_dir'], opt['exp_name'])
timestamp = datetime.now().strftime("%Y%h%d%H%M")
if os.path.exists(result_dir):
    os.system(f'mv {result_dir} {result_dir}_archived_{timestamp}')
os.makedirs(result_dir, exist_ok=False)
os.system(f'cp {config_path} {result_dir}')
os.system(f'cp {ckpt_path} {result_dir}')

model_type = opt['model_type']
# get model dynamically
model = getattr(importlib.import_module('models'), model_type)(opt)
model.inference()

print('[JZSHERLOCK] all done, everything ok')
