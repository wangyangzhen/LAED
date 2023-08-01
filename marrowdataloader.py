"""Pytorch dataset object that loads MNIST dataset as bags."""

from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms
from PIL import Image
import GlobalManager as gm

path=gm.get_value('path')


normalize = transforms.Normalize(
    mean=[0.15, 0.35, 0.35],
    std=[0.25, 0.26, 0.28]
)    

train_preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # normalize,
]) 

test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
]) 

  
def MarrowLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_tensor = train_preprocess(img_pil)

    return img_tensor
def TestLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_tensor = test_preprocess(img_pil)
    return img_tensor
class trainset(Dataset):
    def __init__(self, loader=MarrowLoader):
        self.loader = loader
        self.rd=np.random.RandomState()
    def __getitem__(self, index):
        train_folder=gm.get_value('train_folder')
        target=0
        im_list=glob.glob(os.path.join(train_folder,'*.tif'))
        index=np.random.randint(0,len(im_list),1)[0]
        fn = im_list[index]
        img = self.loader(fn)
        return img,target
    def __len__(self):
        return 100000

class testset(Dataset):
    def __init__(self, loader=TestLoader):
        self.loader = loader
        self.rd=np.random.RandomState()
        train_folder=gm.get_value('train_folder')
        self.test_image=glob.glob(os.path.join(train_folder,'*.tif'))
        # print(len(self.test_image))
        # print(self.test_image)
    def __getitem__(self, index):
        target=0
        fn = self.test_image[index]
        img = self.loader(fn)
        return img,target
    def __len__(self):
        # gm.set_value("test_image",self.test_image)
        return len(self.test_image)
    
