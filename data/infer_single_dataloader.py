import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json
import cv2
import pandas as pd
from glob import glob


class InferSingleDataset(Dataset):
    """
        Images for inference, usually in single folder
            test_images/
                test_1.png
                test_2.png
                ...
        if not in same folder, assign location using csv file
        Returns:
            {"img": img, "img_path": img_path}
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot = opt_dataset['dataroot']
        self.csv_path = opt_dataset['csv_path'] if 'csv_path' in opt_dataset else None
        imgpath_colname = opt_dataset['imgpath_colname']
        is_append_root = opt_dataset['is_append_root']
        augment_opt = opt_dataset['augment']
        augment_type = augment_opt.pop('augment_type')
        self.augment = importlib.import_module(f'data_augment.{augment_type}').val_augment(**augment_opt)
        # used if OOD class (train/val subdir not complete classes)
        json_path = opt_dataset['inv_classmap_json']
        with open(json_path, 'r') as f_json:
            self.inv_classmap = json.load(f_json)

        if self.csv_path is not None:
            self.df = pd.read_csv(self.csv_path)
            img_paths = self.df[imgpath_colname]
            if is_append_root:
                self.img_paths = [os.path.join(self.dataroot, relpath) for relpath in img_paths]
            else:
                self.img_paths = img_paths
        else:
            self.img_paths = [os.path.join(self.dataroot, relpath) for relpath in os.listdir(self.dataroot)]

    def __getitem__(self, index):
        cur_img_path = self.img_paths[index]
        cur_img = cv2.imread(cur_img_path)
        img_aug = self.augment(image=cur_img)['image']
        # simple dataset cannot cover mixup/contrast etc. which need 2 or more images to return
        output_dict = {
            "img" : img_aug,
            "img_path": cur_img_path
        }
        return output_dict

    def __len__(self):
        return len(self.img_paths)


def InferSingleDataloader(opt_dataloader):
    batch_size = 1
    num_workers = 0
    shuffle = False
    infer_dataset = InferSingleDataset(opt_dataloader)
    dataloader = DataLoader(infer_dataset, batch_size=batch_size, pin_memory=True, \
                            drop_last=True, shuffle=shuffle, num_workers=num_workers)
    return dataloader, infer_dataset.inv_classmap
