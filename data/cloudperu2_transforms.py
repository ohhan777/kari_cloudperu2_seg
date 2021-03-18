import albumentations as A
import albumentations.pytorch import ToTensorV2

class ImageAug:
    def __init__(self):
        self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                             A.VerticalFlip(p=0.5),
                             A.ShiftScaleRotate(p=0.5),
                             A.RandomBrightnessContrast(p=0.3),
                             ToTensorV2()])

    def __call__(self, img, mask_img):
        transformed = self.aug(image=img, mask=mask_img)
        return transformed['image'], transformed['mask']

def get_transforms():
    transforms = ImageAug()
    return transforms