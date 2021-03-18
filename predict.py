import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import geotiff_read, cv2_imshow
import argparse
from torchvision import models


def predict(opt):   
    # model  
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=2)
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # GPU-support
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load weights
    assert os.path.exists(opt.weight), "no found the model weight"
    checkpoint = torch.load(opt.weight)
    model.load_state_dict(checkpoint['model'])

    # input
    input_files = []
    imgs = torch.tensor([])
    for input_file in opt.input:
        input_files.append(input_file)
        img = geotiff_read(input_file)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()/255.0 # tensor in [0,1]
        imgs = torch.cat([imgs, img.unsqueeze(0)])  # image stacking (like a batch)
    
    model.eval()
    imgs.to(device)
    with torch.no_grad():
        preds = model(imgs)['out']  # (B, C, H, W)
        preds = torch.argmax(preds, axis=1) # (B, H, W)
        for i in range(len(preds)):   
            cv2_imshow(imgs[i], preds[i])     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='+',
                        help='input images', required=True)
    parser.add_argument('--weight', '-w', type=str, default='weights/cloudperu2_deeplabv3_best.pth',
                        help='weight file path')
    opt = parser.parse_args()

    predict(opt)
