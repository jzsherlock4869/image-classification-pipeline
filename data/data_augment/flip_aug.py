import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_augment(size):
    aug = A.Compose([
        A.Resize(size, size),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
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