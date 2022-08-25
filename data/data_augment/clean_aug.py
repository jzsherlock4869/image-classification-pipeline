import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_augment(size=256):
    aug = A.Compose([
        A.Resize(size, size),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
        ToTensorV2(),
    ])
    return aug

def val_augment(size):
    aug = A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
        ToTensorV2(),
    ])
    return aug