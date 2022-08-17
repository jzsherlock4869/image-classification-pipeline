import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
import pandas as pd
from glob import glob


class SimpleCSVDataset(Dataset):
    """
        Images with class labels are stored in csv file, where:
            img_path,            class_name
            /path/to/img1.png,      cat
            /path/to/img2.png,      dog
            ...
        Returns:
            self.classmap: dict as class_name-class_id mapping
            {"img": img, "class_id": cls_id}
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot = opt_dataset['dataroot']
        self.csv_path = opt_dataset['csv_path']
        imgpath_colname = opt_dataset['imgpath_colname']
        is_append_root = opt_dataset['is_append_root']
        label_colname = opt_dataset['label_colname']
        self.phase = opt_dataset['phase']
        augment_opt = opt_dataset['augment']
        augment_type = augment_opt.pop('augment_type')
        if self.phase == 'train':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').train_augment(**augment_opt)
        elif self.phase == 'val':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').val_augment(**augment_opt)
        # used if OOD class (train/val subdir not complete classes)
        predefined_classmap = opt_dataset['predefined_classmap']

        self.df = pd.read_csv(self.csv_path)
        img_paths = self.df[imgpath_colname]
        if is_append_root:
            self.img_paths = [os.path.join(self.dataroot, relpath) for relpath in img_paths]
        else:
            self.img_paths = img_paths
        label_names = self.df[label_colname]

        if not predefined_classmap:
            class_names = sorted(list(set(label_names)))
            self.classmap = dict([(v, k) for k, v in enumerate(class_names)])  # {'class_1': 0}
            self.inv_classmap = dict([(k, v) for k, v in enumerate(class_names)])  # {0: 'class_1'}
        else:
            class_names = list(predefined_classmap.keys())
            self.classmap = predefined_classmap  # {'class_1': 0}
            self.inv_classmap = dict([(k, v) for v, k in predefined_classmap.items()])  # {0: 'class_1'}
    
        self.labels = [self.classmap[label_name] for label_name in label_names]

    def __getitem__(self, index):
        cur_img_path = self.img_paths[index]
        cur_label = self.labels[index]
        cur_img = cv2.imread(cur_img_path)
        img_aug = self.augment(image=cur_img)['image']
        # simple dataset cannot cover mixup/contrast etc. which need 2 or more images to return
        output_dict = {
            "img" : img_aug,
            "label": cur_label,
            "img_path": cur_img_path
        }
        return output_dict

    def __len__(self):
        return len(self.img_paths)


def SimpleCSVDataloader(opt_dataloader):
    phase = opt_dataloader['phase']
    if phase == 'train':
        batch_size = opt_dataloader['batch_size']
        num_workers = opt_dataloader['num_workers']
        shuffle = True
    elif phase == 'val':
        batch_size = 1
        num_workers = 0
        shuffle = False
    folder_dataset = SimpleCSVDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=batch_size, pin_memory=True, \
                            drop_last=True, shuffle=shuffle, num_workers=num_workers)
    inv_classmap = folder_dataset.inv_classmap
    return dataloader, inv_classmap
