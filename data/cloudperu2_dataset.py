import os
import torch
import glob
from utils.utils import geotiff_read
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CloudPeru2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        if train:
            self.img_dir = self.root/Path('cloudperu2/train/images')
        else:
            self.img_dir = self.root/Path('cloudperu2/val/images')
        self.img_files = sorted(glob.glob(os.path.join(str(self.img_dir),'*.tif')))
        self.transforms = get_transforms(train)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = img_file.replace('/images','/masks').replace('.tif', '_Mask.tif')
        img = geotiff_read(img_file, 4)
        mask_img = geotiff_read(mask_file, 1)
        # transforms (numpy to tensor)
        img, mask_img = self.transforms(img, mask_img)
        return img, mask_img

    def __len__(self):
        return len(self.img_files)


class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                 A.VerticalFlip(p=0.5),
                                 A.ShiftScaleRotate(p=0.5),
                                 A.RandomBrightnessContrast(p=0.3),
                                 ToTensorV2()])
        else:
            self.aug = ToTensorV2()

    def __call__(self, img, mask_img):
        transformed = self.aug(image=img, mask=np.squeeze(mask_img))
        return transformed['image']/255.0, transformed['mask']/255.0

def get_transforms(train):
    transforms = ImageAug(train)
    return transforms