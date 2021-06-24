# dataset and dataloader construction
# support both images (jpg/png/tif etc.) and numpy array (npy) file

from operator import pos
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from basic_config import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_img(path):
    im_bgr = cv2.imread(path)
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


def get_npy(path):
    img = np.load(path)
    return img


class CustomDataset(Dataset):
    """
    Custom dataset for image (in common format or .npy) classification
    """
    def __init__(self, df, data_root,
                transforms=None,
                output_label=True,
                task_name='lbl',
                postfix='npy',
                num_class=2,
                mapp=None):
        """
        args:
            df (pandas.Dataframe) : columns ['imid', 'label_1', 'label_2', ... ], imageid must named 'imid'
            data_root (str) : where images or npys are located
            transforms (local_albumentations.BasicTransform) : data augmentation and transformations
            output_label (bool) : return (img, label) or img only, differ when train and test
            task_name (str) : column key in df to refer to target task label, e.g. 'label_1'
            postfix (str) : postfix of image or npy file, e.g. 'jpg' 'png' 'npy'
        """
        super(CustomDataset).__init__()

        if output_label:
            assert task_name in df.keys()
            # [delete] drop no label samples in target task name
            # self.df = df.dropna(subset=[task_name]).reset_index(drop=True).copy()
            assert not df[task_name].isnull().values.any(), 'label with nan should be droped before'
        
        self.df = df
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.postfix = postfix
        if output_label:
            self.labels = self.df[task_name].values
            assert len(np.unique(self.labels)) == num_class, 'label file has more class number than config'
            if mapp is not None:
                self.labels = list(map(lambda x: mapp[x], self.labels))


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels if train/val
        if self.output_label:
            target = self.labels[index]
            img_path = os.path.join(self.data_root, '{}.{}'.format(self.df.loc[index]['imid'], self.postfix))
        else:
            img_path = os.path.join(self.data_root, '{}.{}'.format(self.df.loc[index]['imid'], self.postfix))

        if self.postfix == 'npy':
            img = get_npy(img_path)
        else:
            img = get_img(img_path)

        # not support dual transform (like mixup augment)
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.output_label:
            return img, target
        else:
            return img


def get_train_transforms_img(config, force_light=False):
    """
    basic transforms for image (can be processed by cv2) input for train
    """
    aug_list=[
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, val_shift_limit=5, p=0.3),
        A.Resize(height=config.img_size_h, width=config.img_size_w),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)


def get_valid_transforms_img(config):
    """
    basic transforms for image (can be processed by cv2) input for valid
    """
    aug_list=[
        A.Resize(height=config.img_size_h, width=config.img_size_w),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)


def get_train_transforms_npy(config, force_light=False):
    """
    basic transforms for npy array input for train
    mean and std should be set in config
    """
    norm_mean = config.norm_mean
    norm_std = config.norm_std

    if norm_mean is None or norm_std is None:
        print('dataset norm in train not provided,'\
            ' using default mean = 0, std = 1 (may cause train failure)')
        norm_mean = [0.0 for _ in range(config.num_input_chs)]
        norm_std = [1.0 for _ in range(config.num_input_chs)]
    
    aug_list=[
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.3),
        A.Resize(height=config.img_size_h, width=config.img_size_w),
        A.NormalizeMultispectral(
            mean=norm_mean,
            std=norm_std,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)


def get_valid_transforms_npy(config):
    """
    basic transforms for npy array input for valid
    mean and std should be set in config
    """
    norm_mean = config.norm_mean
    norm_std = config.norm_std

    if norm_mean is None or norm_std is None:
        print('dataset norm in valid not provided,'\
            ' using default mean = 0, std = 1 (may cause valid failure)')
        norm_mean = [0.0 for _ in range(config.num_input_chs)]
        norm_std = [1.0 for _ in range(config.num_input_chs)]
    
    aug_list=[
        A.Resize(height=config.img_size_h, width=config.img_size_w),
        A.NormalizeMultispectral(
            mean=norm_mean,
            std=norm_std,
            p=1.0),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(aug_list, p=1.0)



def get_dataset_with_folds(task_name):
    """
    split trainval image list into train/val in K fold manner
    return:
        trainval_df_nona (pandas.Dataframe) : trainval df for split
        folds (list of tuples) : (trn_idx, val_idx), index in trainval_df_nona
    """
    trainval_df = pd.read_csv(config.TRAIN_CSV)
    # select task id here
    trainval_df_nona = trainval_df.dropna(subset=[task_name]).reset_index(drop=True).copy()
    skf = StratifiedKFold(n_splits=config.fold_num,
                            shuffle=True, random_state=config.seed
                            )
    folds = skf.split(np.arange(trainval_df_nona.shape[0]), trainval_df_nona[config.task_name].values)
    return trainval_df_nona, folds


def prepare_dataloader(df, trn_idx, val_idx, config):

    data_root=config.TRAIN_IMAGE_DIR
    task_name = config.task_name
    postfix = config.postfix
    num_class = config.num_class
    classname_id_map = config.classname_id_map

    df_nona = df.dropna(subset=[task_name]).reset_index(drop=True).copy()
    train_ = df_nona.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df_nona.loc[val_idx, :].reset_index(drop=True)

    if postfix == 'npy':
        tsfm_train = get_train_transforms_npy(config)
        tsfm_valid = get_valid_transforms_npy(config)
    else:
        tsfm_train = get_train_transforms_img(config)
        tsfm_valid = get_valid_transforms_img(config)
    
    train_ds = CustomDataset(train_, data_root, transforms=tsfm_train,
                              output_label=True, task_name=task_name, 
                              postfix=postfix, num_class=num_class, mapp=classname_id_map)
    valid_ds = CustomDataset(valid_, data_root, transforms=tsfm_valid,
                              output_label=True, task_name=task_name, 
                              postfix=postfix, num_class=num_class, mapp=classname_id_map)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train_bs,
        drop_last=True,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=config.valid_bs,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader
