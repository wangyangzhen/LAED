"""
Created on Wed Aug 12 09:49:46 2020

@author: yangzhen
"""
#-*- coding:utf-8 -*-
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

os.environ['KMP_DUPLICATE_LIB_OK']='True'
detectron2_repo_path = r"D:/programming/detectron2-0.6" #自己的detectron2所在

mydata_metadata = MetadataCatalog.get("marrow_seg")
cfg = get_cfg()
cfg.merge_from_file(
    # os.path.join(detectron2_repo_path, "D:/WYZ/TYL/detectron2-master/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    os.path.join(detectron2_repo_path,"configs//COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)  

cfg.DATASETS.TEST = ()  
cfg.DATALOADER.NUM_WORKERS = 0


cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MAX_SIZE_TEST=1280
cfg.INPUT.MIN_SIZE_TEST=720

cfg.MODEL.WEIGHTS = os.path.join('D:/programming/Leukemia_Diagnosis-main/cell_seg_20211229', "model_0005999.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

cfg.DATASETS.TEST = ("data", )
predictor = DefaultPredictor(cfg)
import itertools
from detectron2.utils.visualizer import ColorMode
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import json
from PIL import Image
from torchvision import  transforms
#%%
model = torch.load('D:/data/bone_marrow/mil_output/indep_models/res_mat_618/1-3_validate/k_indep_13_5.pth')
model.cuda()
#%% 
data_path='D:/data/bone_marrow/dataset/test'
output_path='D:/data/bone_marrow/dataset/attmap'
normalize = transforms.Normalize(
    mean=[0.15, 0.35, 0.35],
    std=[0.25, 0.26, 0.28]
)    
test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
]) 
d_list=os.listdir(data_path)
for fd in d_list:
    pred_path=os.path.join(data_path,fd)
    output_fd=os.path.join(output_path,fd)
    os.mkdir(output_fd)
    c=0
    pred_list=os.listdir(pred_path) 
    # img_bag=torch.zeros([0,3,224,224])
    for pred_im in pred_list:
        # pred_im='18-9-7-21.tif'
        file=os.path.splitext(pred_im)
        filename,type=file
        if type!='.tif':
            continue
        im = cv2.imread(os.path.join(pred_path,pred_im))
        im_y,im_x,_=im.shape
        outputs = predictor(im)
        instance=outputs['instances'].to("cpu")._fields
        pred_box=instance['pred_boxes']
        pred_score=instance['scores']
        mask=np.zeros(outputs['instances'].to("cpu")._image_size)
        for idx in range(len(pred_score)):
            if pred_score[idx]<0.9:
                continue
            box_1=[pred_box.tensor.numpy()[idx][0], pred_box.tensor.numpy()[idx][1],pred_box.tensor.numpy()[idx][2], pred_box.tensor.numpy()[idx][3]]
            mask=instance['pred_masks']
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
            img_pil = Image.fromarray(cv2.cvtColor(np.uint8(mask_patch),cv2.COLOR_BGR2RGB))
            img_pil =img_pil.resize((224,224))
            img_tensor = test_preprocess(img_pil) 
            img_tensor = img_tensor[np.newaxis, :, :, :]
            # img_bag=torch.stack((img_bag, img_tensor), 0)
            if c==0:
                img_bag=img_tensor
                c+=1
            else:
                img_bag=torch.cat((img_bag, img_tensor), 0)
            # path=os.path.join(output_path,lable,filename+str(idx)+'.tif')
    #%
    with torch.no_grad():
        output_sum=model(img_bag.cuda())
        pred_type=output_sum[2][0].cpu().numpy()[0]
        att=output_sum[3][:,pred_type].cpu().numpy()
        att=att-att.min()
        att=np.uint8((att/att.max()*255))
    #% 
    print(fd+' : '+str(pred_type))
    c=0
    
    for pred_im in pred_list:
        # pred_im='18-9-7-21.tif'
        file=os.path.splitext(pred_im)
        filename,type=file
        if type!='.tif':
            continue
        im = cv2.imread(os.path.join(pred_path,pred_im))
        im_y,im_x,_=im.shape
        outputs = predictor(im)
        instance=outputs['instances'].to("cpu")._fields
        pred_box=instance['pred_boxes']
        pred_score=instance['scores']
        mask_sum=instance['pred_masks']
        rgb_mask=np.zeros(im.shape)
        for idx in range(len(pred_score)):
            if pred_score[idx]<0.9:
                continue
            mask=np.uint8(mask_sum[idx,:,:].numpy())
            mask=cv2.dilate(mask,np.ones((30, 30), np.uint8))-cv2.dilate(mask,np.ones((20,20), np.uint8))
            CM=np.array(plt.colormaps.get_cmap('jet')(att[c]))
            ColorMap=np.uint8(CM*254)+1
            rgb_mask[:,:,0]=np.where(mask>0,ColorMap[0],rgb_mask[:,:,0])
            rgb_mask[:,:,1]=np.where(mask>0,ColorMap[1],rgb_mask[:,:,1])
            rgb_mask[:,:,2]=np.where(mask>0,ColorMap[2],rgb_mask[:,:,2])
            rgb_mask=np.uint8(rgb_mask)
            c+=1
        im_c=np.where(rgb_mask>0,rgb_mask,im)
        cv2.imwrite(os.path.join(output_fd,pred_im),im_c)
#%%
im_c=np.where(rgb_mask>0,rgb_mask,im)
cv2.imshow('image',im_c)

cv2.waitKey(0)
cv2.destroyWindow('image')