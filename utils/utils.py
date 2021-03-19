import numpy as np
from osgeo import gdal
import cv2
import glob
import torch

def geotiff_read(filename, num_bands=4):
    ds = gdal.Open(filename)
    if ds == None:
        print("Error loading %s." % filename)
        return
    band = []
    for i in range(num_bands):
        band.append(ds.GetRasterBand(i + 1).ReadAsArray().astype(np.uint8))
        img = np.dstack(band)
    return img


def cv2_imshow(img, mask_img=None, window_title='img'):
   num_channels = 3 if img.shape[2] > 3 else img.shape[2] 
   if torch.is_tensor(img):
       img = img.mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # (C,H,W) to (H,W,C)
       # img = img[...,::-1].astype(np.uint8) # (R,G,B) to (B,G,R)

   output_img = img[:,:,:num_channels]
   if mask_img is not None:
       if torch.is_tensor(mask_img):
           mask_img = mask_img.mul(255).cpu().numpy().astype(np.uint8)
       mask_img = np.squeeze(mask_img) # (512, 512, 1)>(512, 512)
       mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
       mask_img[:, :, 0] = 0 # yellow
       output_img = cv2.addWeighted(mask_img, 0.3, output_img, 1, 0)
   cv2.imshow(window_title, output_img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

img_files = glob.glob('../data/cloudperu2/val/images/*.tif')
for img_file in img_files:
    mask_file = img_file.replace('/images', '/masks').replace('.tif','_Mask.tif')
    print(mask_file)
    img = geotiff_read(img_file, 4) 
    mask_img = geotiff_read(mask_file, 1)
    cv2_imshow(img, mask_img)



def fitness_test(preds, targets):
    # IoU
    intersection = (preds & targets).float().sum((1, 2))  
    union = (preds | targets).float().sum((1, 2))        
    iou = (intersection + 1e-6) / (union + 1e-6)
    pix_accuracy = (preds == targets).float().sum((1,2))/preds[0].numel()  
    
    return iou, pix_accuracy
     
