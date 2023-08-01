# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 01:06:30 2022

@author: Administrator
"""

from __future__ import print_function

import numpy as np
import os
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import scipy.io as scio

# from marrowdataloader import MarrowBags
# from MarrowClasLoader import trainset,testset
from model_test import Attention, GatedAttention
import GlobalManager as gm
train_path='D:/data/bone_marrow/training/bag_class_k4'
type_list=os.listdir(os.path.join(train_path,'train'))
# train_path='D:/data/bone_marrow/training/test'
gm._init_()
gm.set_value("path",train_path)
from marrowdataloader import testset
import torch.nn as nn
from tqdm import tqdm 
# gm.set_value("gl",batch_idx)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1*10e-5, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--bag_length', type=int, default=128, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=0, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=1, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=8, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


print('Init Model')
test_loader = data_utils.DataLoader(testset(),
                                     batch_size=args.bag_length,
                                     shuffle=True,
                                     **loader_kwargs)  
#%%

score_th=[0.99,0.92,0.97,0.93,0.91]
for si in range(0,5):
    model = torch.load('D:/data/bone_marrow/mil_output/models/b64_k'+str(si+1)+'.pth')
    if args.cuda:
        model.cuda()
    while(1):
        train_path='D:/data/bone_marrow/training/bag_class_k'+str(si+1)
        f_output='D:/data/bone_marrow/mil_output/'
        cm=np.zeros([5,5])
        repeat=10
        # folder_idx=0
        pred_type='test'
        cm=np.zeros([5,5])
        rep_record=np.zeros([repeat,1])
        score_record=np.zeros([repeat,1])
        folder_list=os.listdir(os.path.join(train_path,pred_type))
        test_list=[]
        acc=0
        fn=0
        bag_label=-1
        with torch.no_grad():
            for folder in folder_list:
                bag_label+=1
                test_list=os.listdir(os.path.join(train_path,pred_type,folder))
                for file in test_list:
                    with tqdm(total=repeat) as t: 
                        test_folder=os.path.join(train_path,pred_type,folder,file)
                        gm.set_value("train_folder",test_folder)
                        count=0
                        tp=0
                        ouput_record=np.zeros([repeat,1])-1
                        score_out=np.zeros([repeat,1])
                        for batch_idx, (data, label) in enumerate(test_loader):
                            # print(batch_idx)
                            # bag_label = label
                            if bag_label==1:
                                target=torch.tensor([1]).long()
                            elif bag_label==2:
                                target=torch.tensor([2]).long()
                            elif bag_label==3:
                                target=torch.tensor([3]).long()
                            elif bag_label==4:
                                target=torch.tensor([4]).long()
                            else:
                                target=torch.tensor([0]).long()
                            if args.cuda:
                                data = data.cuda()
                            data = Variable(data)
                            # data = data.squeeze(0)
                            
                            output_sum=model(data)
                            output=output_sum[0]
                            pred=output_sum[2][0].cpu().numpy()[0]
                            ouput_record[count,0]=pred
                            # print(count)
                            ouput_record=np.uint8(ouput_record)    
                            counts = np.bincount(ouput_record[:,0])
                            pred_res=np.argmax(output.cpu().numpy()[0])
                            scores=output.cpu().numpy()[0]
                            scores.sort()
                            # print(scores)
                            score_out[count,0]=scores[4]-scores[3]
                            count+=1
                            if pred_res==target.cpu().numpy()[0]:
                                tp+=1
                            t.set_description('Iteration %i' % fn)
                            t.set_postfix(acc=tp/count)
                            t.update(1)
                            if count>=repeat:
                                break
                        # print(test_folder)
                        # print(tp/repeat)
                        if tp/repeat>0.5:
                            acc+=1
                        cm[bag_label,np.argmax(counts)]+=1
                        ouput_record=ouput_record-bag_label
                        rep_record=np.concatenate((rep_record,ouput_record),1)
                        score_record=np.concatenate((score_record,score_out),1)
                        fn+=1
            # print(acc/fn)   
            if acc/fn>=score_th[si]-1:
                print('acc: '+str(acc/fn))
                scio.savemat(f_output+'bg_'+str(args.bag_length)+'_k_'+str(si+1)+'_20220409.mat', {'cm':cm,'acc':acc/fn,'rep_record':rep_record,'score_record':score_record})
                t.close()
                break
        # torch.cuda.empty_cache()