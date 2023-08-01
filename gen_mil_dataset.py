# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:30:31 2021

@author: yangzhen
"""

import os
import xlrd
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

detectron2_repo_path = r"D:/program/detectron2-0.6" #自己的detectron2所在
img_path = r"D:/data/bone_marrow/training/train_seg/train"
json_path = r"D:/data/bone_marrow/training/train_seg/train_seg.json"

      
# register_coco_instances("marrow", {"thing_colors":[np.array([0, 0, 1]),np.array([1, 0, 1]),np.array([1, 0, 0]),np.array([1, 1, 0]),np.array([0,1, 1])]}, json_path, img_path)
register_coco_instances("marrow_seg", {"thing_colors":[[0, 0, 255],[255, 0, 0],[0, 255, 0],[255, 255, 0],[0,255, 255],[255, 0, 255],[0 ,122, 122],
                                                   [122, 0, 122],[125, 122, 0],[0, 122, 0],[122, 0, 0],[0,0,122],[122, 0, 255],[255 ,122, 0],[122 ,122,122]]}, json_path, img_path)
# register_coco_instances("marrow",{}, json_path, img_path)
mydata_metadata = MetadataCatalog.get("marrow_seg")
dataset_dicts = DatasetCatalog.get("marrow_seg")


cfg = get_cfg()
cfg.merge_from_file(
    # os.path.join(detectron2_repo_path, "D:/WYZ/TYL/detectron2-master/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    os.path.join(detectron2_repo_path, "D:/program/detectron2-0.6/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)  
#跟下面的cfg.MODEL.WEIGHTS对应，在https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md下面找自己所需要的模型，
cfg.DATASETS.TRAIN = ("marrow_seg",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
# cfg.MODEL.WEIGHTS = r"D:/WYZ/TYL/detectron2-master/model/retina_resnet_101.pkl"  # initialize from model zoo
cfg.MODEL.WEIGHTS = r"D:/program/detectron2-0.6/tools/cell_seg_20211229/model_0005999.pth"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MAX_SIZE_TEST=1280
cfg.INPUT.MIN_SIZE_TEST=720
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

wb = xlrd.open_workbook(r'D:/data/bone_marrow/AML_subtype.xlsx')
table = wb.sheet_by_name('Sheet1')
subtype={}
type_count={'AML':0,'APL':0,'CML':0,'ALL':0,'CLL':0}
for i in range(table.nrows-1):
    x=table.cell(i,1).value
    y=table.cell(i,0).value
    
    if x in 'M3':
        type_count['APL']+=1
        subtype[y]='APL'
    elif x in 'CML':
        type_count['CML']+=1
        subtype[y]='CML'
    elif x in 'ALL':
        type_count['ALL']+=1
        subtype[y]='ALL'
    elif x in 'CLL':
        type_count['CLL']+=1 
        subtype[y]='CLL'
    else:
        type_count['AML']+=1
        subtype[y]='AML'
#%%
path = "D:/data/bone_marrow/dataset_20220423/"
output_path = "D:/data/bone_marrow/training/bag_dataset_20220423/"
for file in os.listdir(path):
    print(subtype[file])
try:
    os.mkdir(os.path.join(output_path,'APL'))
    os.mkdir(os.path.join(output_path,'AML'))
    os.mkdir(os.path.join(output_path,'CLL'))
    os.mkdir(os.path.join(output_path,'CML'))
    os.mkdir(os.path.join(output_path,'ALL'))
except:
    pass
#%%
for folder in os.listdir(path): #该文件夹下所有的文件（包括文件夹）
    print(subtype[folder])
    count=0
    for file in os.listdir(path+folder):   #遍历所有文件
        try:
            os.mkdir(os.path.join(output_path,subtype[folder],folder))
        except:
            pass
        Olddir=os.path.join(path+folder,file)   #原来的文件路径
        filename=os.path.splitext(file)[0]   #文件名
        filetype=os.path.splitext(file)[1]   #文件扩展名
        if filetype != '.tif':
            continue
        im = cv2.imread(os.path.join(path+folder,file))
        im_y,im_x,_=im.shape
        outputs = predictor(im)
        instance=outputs['instances'].to("cpu")._fields
        pred_box=instance['pred_boxes']
        pred_score=instance['scores']
        # mask=np.zeros(outputs['instances'].to("cpu")._image_size)

        for idx in range(len(pred_score)):
            if pred_score[idx]<0.5:
                continue
            box_1=[pred_box.tensor.numpy()[idx][0], pred_box.tensor.numpy()[idx][1],pred_box.tensor.numpy()[idx][2], pred_box.tensor.numpy()[idx][3]]
            mask=instance['pred_masks']
            # mask=instance['pred_masks']
            mask=mask[idx,:,:][:,:,np.newaxis]
            mask=np.tile(mask,(1,1,3))
            kernel = np.ones((25, 25), np.uint8)
            mask = cv2.dilate(np.uint8(mask), kernel, iterations=1)
            seg_im=im*mask
            box_1=np.int32(box_1)
            patch_raw=seg_im[max(box_1[1]-20,0):min(box_1[3]+20,im_y),max(box_1[0]-20,0):min(box_1[2]+20,im_x),:]
            [height,width,]=patch_raw.shape[0:2]
            mask_patch=np.zeros([224,224,3])
            if np.max([width,height])>224:
                patch_raw=cv2.resize(patch_raw,(int(width*224/np.max([width,height])),int(height*224/np.max([width,height]))))
            [height,width]=patch_raw.shape[0:2]
            integ_y=int((224-height)/2)
            integ_x=int((224-width)/2)
            mask_patch[integ_y:integ_y+height,integ_x:integ_x+width,:]=patch_raw
            mask_patch=np.uint8(mask_patch)
            output_name=os.path.join(output_path,subtype[folder],folder,filename+str(idx)+'.png')
            cv2.imwrite(output_name,mask_patch)

#%%
o_path='D:/data/bone_marrow/training/bag_class/test/CLL/'
for folder in os.listdir(o_path): #该文件夹下所有的文件（包括文件夹）
    print(folder)
    count=0
    print(len(os.listdir(o_path+folder)))   #遍历所有文件
#%% train-test segmentation
import shutil

ratio=0.8
dataset_path='D:/data/bone_marrow/training/bag_dataset_20220423/'
train_path='D:/data/bone_marrow/training/bag_class_20220423/bag_class/'
try:
    os.mkdir(os.path.join(train_path,'train'))
    os.mkdir(os.path.join(train_path,'test'))
except:
    pass
type_list=os.listdir(dataset_path)
for type_file in type_list:
    try:
        os.mkdir(os.path.join(train_path,'train',type_list))
        os.mkdir(os.path.join(train_path,'test',type_list))
    except:
        pass
    folder_list=os.listdir(dataset_path+type_file)
    train_num=int(np.ceil(len(folder_list)*ratio))
    train_idx=np.arange(len(folder_list))
    np.random.shuffle(train_idx)
    for i in train_idx[0:train_num]:
        shutil.copytree(os.path.join(dataset_path,type_file,folder_list[i]), os.path.join(train_path,'train',type_file,folder_list[i]))
        print(0)
    for i in train_idx[train_num:]:
        if not(folder_list[i] in os.listdir(os.path.join(train_path,'train',type_file))):
            shutil.copytree(os.path.join(dataset_path,type_file,folder_list[i]), os.path.join(train_path,'test',type_file,folder_list[i]))     
            # print(folder_list[i])
    