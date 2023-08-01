
"""
Created on Wed Aug 12 09:49:46 2020

@author: yangzhen
"""
#%% seg cell by train_p folder

import albumentations as A
import glob
import os
import numpy as np
import cv2
import random
import shutil

aug_num=30000
train_rate=1
dataset="D:/data/bone_marrow/training/class"
trainset="D:/data/bone_marrow/training/bag_class"
try:
    os.mkdir(os.path.join(trainset,'train'))
    os.mkdir(os.path.join(trainset,'test'))   
except:
    pass
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(always_apply=False, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, brightness_by_max=True, always_apply=False, p=1),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.GaussNoise (var_limit=(20.0, 100.0), mean=0, always_apply=False, p=0.5),
    A.CoarseDropout(max_holes=10, max_height=10, max_width=10, min_holes=None, min_height=None, min_width=None, fill_value=0, always_apply=False, p=0.1),
    A.RGBShift(r_shift_limit=20, g_shift_limit=0, b_shift_limit=20, always_apply=False, p=0.5),
    A.ISONoise (color_shift=(0.005, 0.01), intensity=(0.05, 0.3), always_apply=False, p=0.5),
    A.MultiplicativeNoise (multiplier=(0.95, 1.05), per_channel=False, elementwise=False, always_apply=False, p=0.2),
    A.Downscale (scale_min=0.8, scale_max=0.99, interpolation=0, always_apply=False, p=0.2)
    ])
for fn in os.listdir(dataset):
    os.mkdir(os.path.join(trainset,'train',fn))
    os.mkdir(os.path.join(trainset,'test',fn))
    data_num=len(os.listdir(os.path.join(dataset,fn)))
    count=data_num
    for imn in os.listdir(os.path.join(dataset,fn)):
        count-=1
        fnn,_=os.path.splitext(imn)
        print(fn+':'+str(count))
        if train_rate<np.random.rand(1):
            
            # shutil.copy(os.path.join(dataset,fn,imn), os.path.join(trainset,'test',fn,imn))
            im=cv2.imread(os.path.join(dataset,fn,imn))
            cv2.imwrite(os.path.join(trainset,'test',fn,fnn+"_{:04d}".format(0)+'.png'),im)
            # data_num-=1
            continue
        aug_rate=int(np.floor(aug_num/data_num))
        shutil.copy(os.path.join(dataset,fn,imn), os.path.join(trainset,'train',fn,imn))
        fcount=1
        im=cv2.imread(os.path.join(dataset,fn,imn))
        for i in range(aug_rate):
            if aug_num/data_num>np.random.rand(1):
                # im=cv2.imread(os.path.join(dataset,fn,imn))
                ima=transform(image=im.copy())['image']
                cv2.imwrite(os.path.join(trainset,'train',fn,fnn+"_{:04d}".format(fcount)+'.png'),ima)
                fcount+=1