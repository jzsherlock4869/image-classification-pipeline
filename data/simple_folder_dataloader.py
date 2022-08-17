import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from glob import glob


class SimpleFolderDataset(Dataset):
    """
        Images of different classes arranged in different folders:
            dataroot/
                class_1/
                    1_000.png
                    1_001.png
                class_2/
                    2_000.png
                    2_001.png
                ...
        Returns:
            self.classmap: dict as class_name-class_id mapping
            {"img": img, "class_id": cls_id}
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot = opt_dataset['dataroot']
        self.phase = opt_dataset['phase']
        self.postfixs = opt_dataset['postfixs']
        augment_opt = opt_dataset['augment']
        augment_type = augment_opt.pop('augment_type')
        if self.phase == 'train':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').train_augment(augment_opt)
        elif self.phase == 'val':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').val_augment(augment_opt)
        # used if OOD class (train/val subdir not complete classes)
        predefined_classmap = opt_dataset['predefined_classmap']

        if not self.classmap:
            class_names = sorted(os.listdir(self.dataroot))
            self.classmap = dict([(v, k) for k, v in enumerate(class_names)])  # {'class_1': 0}
            self.inv_classmap = dict([(k, v) for k, v in enumerate(class_names)])  # {0: 'class_1'}
        else:
            class_names = list(predefined_classmap.keys())
            self.classmap = predefined_classmap  # {'class_1': 0}
            self.inv_classmap = dict([(k, v) for v, k in predefined_classmap.items()])  # {0: 'class_1'}

        self.img_paths, self.labels = list(), list()
        for class_name in class_names:
            full_dirname = os.path.join(self.dataroot, class_name)
            class_img_paths = []
            for postfix in self.postfixs:
                class_img_paths.append(glob(os.path.join(full_dirname, f'*.{postfix}')))
            # merge to all paths and labels
            self.img_paths.append(class_img_paths)
            self.labels.append([self.classmap[class_name] for _ in range(len(class_img_paths))])
    
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


def SimpleFolderDataloader(opt_dataloader):
    phase = opt_dataloader['phase']
    if phase == 'train':
        batch_size = opt_dataloader['batch_size']
        num_workers = opt_dataloader['num_workers']
        shuffle = True
    elif phase == 'val':
        batch_size = 1
        num_workers = 0
        shuffle = False
    folder_dataset = SimpleFolderDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=batch_size, pin_memory=True, \
                            drop_last=True, shuffle=shuffle, num_workers=num_workers)
    return dataloader
