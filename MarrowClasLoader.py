"""Pytorch dataset object that loads MNIST dataset as bags."""

from __future__ import print_function

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

import argparse

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

# from resnet50_raw import resnet50
import cv2
import os
import glob
from torchvision import transforms, utils
from PIL import Image
import albumentations as A

path="D:/data/bone_marrow/training/BM_class_20220127"

train_path=os.path.join(path,'train')
test_path=os.path.join(path,'test')
train_list=[]
train_label=[]
test_list=[]
test_label=[]
dataset_interval=[]
dataset_interval.extend([0])
train_file_list=os.listdir(train_path)
test_file_list=os.listdir(test_path)
clas=0
count=0
for folder in train_file_list:
    im_list=glob.glob(os.path.join(train_path,folder,'*.jpg'))
    train_list.extend(im_list)
    train_label.extend(np.zeros(len(im_list))+clas)
    count+=len(im_list)
    dataset_interval.extend([count-1])
    clas+=1
    
clas=0
count=0   
for folder in test_file_list:
    im_list=glob.glob(os.path.join(test_path,folder,'*.jpg'))
    test_list.extend(im_list)
    test_label.extend(np.zeros(len(im_list))+clas)
    count+=len(im_list)
    # dataset_interval.extend([count-1])
    clas+=1
# print(test_label)
normalize = transforms.Normalize(
    mean=[0.36, 0.22, 0.4],
    std=[0.32, 0.28, 0.39]
)    
train_preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    # transforms.RandomFlip(prob=0.5, horiz=True, vert=False),
    # transforms.RandomFlip(prob=0.5, horiz=False, vert=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomCrop(224, padding=32),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
    transforms.ToTensor(),
    normalize
]) 
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
    A.GaussNoise (var_limit=(50.0, 300.0), mean=0, always_apply=False, p=0.2),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.3), always_apply=False, p=0.2),
    A.MultiplicativeNoise (multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.2),
    A.Downscale (scale_min=0.75, scale_max=0.9, interpolation=0, always_apply=False, p=0.2)
    ],)

test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
]) 

def MarrowLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_tensor = train_preprocess(img_pil)
    # img_array = np.array(img_pil)
    # img_array = transform(image=img_array)
    # img_pil = Image.fromarray(np.uint8(img_pil))
    # img_tensor = train_preprocess(img_pil)
    return img_tensor

def TestLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_tensor = test_preprocess(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=MarrowLoader):
        #定义好 image 的路径
        self.images = train_list
        self.target = train_label
        self.loader = loader
        self.dataset_interval=dataset_interval
        self.rd=np.random.RandomState()
    def __getitem__(self, index):
        # index=np.random.randint(0,len(train_list),1)[0]
        # print(index)
        # index = torch.LongTensor(self.rd.randint(self.dataset_interval[file_idx], self.dataset_interval[file_idx+1], 1))
        # print(index)
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target
    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self, loader=TestLoader):
        #定义好 image 的路径
        self.images = test_list
        self.target = test_label
        self.loader = loader

        self.rd=np.random.RandomState()
    def __getitem__(self, index):
        # index=np.random.randint(0,len(test_list),1)[0]
        # print(index)
        # index = torch.LongTensor(self.rd.randint(self.dataset_interval[file_idx], self.dataset_interval[file_idx+1], 1))
        # print(index)
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target
    
    def __len__(self):
        return len(self.images)
# class MarrowBags(data_utils.Dataset):
#     def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
#         self.target_number = target_number
#         self.mean_bag_length = mean_bag_length
#         self.var_bag_length = var_bag_length
#         self.num_bag = num_bag
#         self.train = train
#         print('5675')
#         self.r = np.random.RandomState(seed)

#         self.num_in_train = count
#         self.num_in_test = 10

#         if self.train:
#             self.train_bags_list, self.train_labels_list = self._create_bags()
#         else:
#             self.test_bags_list, self.test_labels_list = self._create_bags()

#     def _create_bags(self):
#         if self.train:
#             MarrowLoader = data_utils.DataLoader(trainset(),
#                                            batch_size=self.mean_bag_length,
#                                            shuffle=False)
            
#         else:
#             loader = data_utils.DataLoader(datasets.MNIST('../datasets',
#                                                           train=False,
#                                                           download=True,
#                                                           transform=transforms.Compose([
#                                                               transforms.ToTensor(),
#                                                               transforms.Normalize((0.1307,), (0.3081,))])),
#                                            batch_size=self.num_in_test,
#                                            shuffle=False)

#         for (batch_data, batch_labels) in MarrowLoader:
#             print('1111')
#             all_imgs = batch_data
#             all_labels = batch_labels


#         bags_list = []
#         labels_list = []

#         for i in range(self.num_bag):
#             bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
#             if bag_length < 1:
#                 bag_length = 1
#             if self.train:
#                 indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
#             else:
#                 indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
#             print('*********************')
#             print(len(indices))
#             # print(len(all_labels))

#             labels_in_bag = all_labels[indices]

                
#             bags_list.append(all_imgs[indices])
#             labels_list.append(labels_in_bag.long())

#         return bags_list, labels_list

#     def __len__(self):
#         if self.train:
#             return len(self.train_labels_list)
#         else:
#             return len(self.test_labels_list)

#     def __getitem__(self, index):
#         # if self.train:
#         #     bag = self.train_bags_list[index]
#         #     label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
#         # else:
#         #     bag = self.test_bags_list[index]
#         #     label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
#         # print(index)
#         if self.train:
#             bag = self.train_bags_list[index]
#             label = [self.train_labels_list[index], self.train_labels_list[index]]
#         else:
#             bag = self.test_bags_list[index]
#             label =  self.test_labels_list[index]
#         return bag, label


# if __name__ == "__main__":

# train_loader = data_utils.DataLoader(MarrowBags(target_number=9,
#                                                mean_bag_length=10,
#                                                var_bag_length=0,
#                                                num_bag=100,
#                                                seed=1,
#                                                train=True),
#                                      batch_size=1,
#                                      shuffle=True)

# test_loader = data_utils.DataLoader(MnistBags(target_number=9,
#                                               mean_bag_length=10,
#                                               var_bag_length=2,
#                                               num_bag=100,
#                                               seed=1,
#                                               train=False),
#                                     batch_size=1,
#                                     shuffle=False)

# len_bag_list_train = []
# mnist_bags_train = 0
# for batch_idx, (bag, label) in enumerate(train_loader):
#     len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
#     mnist_bags_train += label[0].numpy()[0]
# print('Number positive train bags: {}/{}\n'
#       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
#     mnist_bags_train, len(train_loader),
#     np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

# len_bag_list_test = []
# mnist_bags_test = 0
# for batch_idx, (bag, label) in enumerate(test_loader):
#     len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
#     mnist_bags_test += label[0].numpy()[0]
# print('Number positive test bags: {}/{}\n'
#       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
#     mnist_bags_test, len(test_loader),
#     np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
