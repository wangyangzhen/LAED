# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 01:41:33 2022

@author: Administrator
"""


from __future__ import print_function

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
# from multi_head_resnet import resnext101_32x8d
from MIL_model import resnext101_32x4d


import GlobalManager as gm
import torch.nn as nn
gm._init_()
# gm.set_value("gl",batch_idx)
from marrowdataloader import trainset,testset
from tensorboardX import SummaryWriter
from tqdm import tqdm 
import time
import scipy.io as scio


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1*10e-5, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--bag_length', type=int, default=64, metavar='ML',
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

# train_loader = data_utils.DataLoader(MarrowBags(target_number=args.target_number,
#                                                mean_bag_length=args.mean_bag_length,
#                                                var_bag_length=args.var_bag_length,
#                                                num_bag=args.num_bags_train,
#                                                seed=args.seed,
#                                                train=True),
#                                      batch_size=1,
#                                      shuffle=True,
#                                      **loader_kwargs)




print('Init Model')
# if args.model=='attention':
#     model = Attention()
# elif args.model=='gated_attention':
#     model = GatedAttention()

#%%
# model = torch.load('best_mode.pth')
# bag_size=[1,2,4,8,16,32]
bag_size=[32]
# f_output='H:/data/bone_marrow/mil_output/'
for k in range(3,4):
    train_path='D:/data/bone_marrow/dataset/internal/k'+str(k)
    type_list=os.listdir(os.path.join(train_path,'train'))
    training_loss=np.zeros([args.epochs])
    testing_loss=np.zeros([args.epochs])
    training_time=np.zeros([args.epochs])
    for bn in range(len(bag_size)):
        bag_length=bag_size[bn]
        # print(f_output+'bg_'+str(bag_size[bn])+'k_'+str(k))
        model=resnext101_32x4d(num_classes=21,pretrained=True)
        train_loader = data_utils.DataLoader(trainset(),
                                             batch_size=bag_length,
                                             shuffle=True,
                                             **loader_kwargs)
        model.fc = torch.nn.Linear(2048, 1024, bias=True)
        if args.cuda:
            model.cuda()
        iteration=0
        for fn in os.listdir(os.path.join(train_path,'train')):
            iteration+=len(os.listdir(os.path.join(train_path,'train',fn)))
        optimizer = optim.Adam(model.parameters(), lr=0.00002, betas=(0.9, 0.999), weight_decay=args.reg)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        fc=[]
        type_idx=int(np.random.randint(0,len(type_list),1))
        folder_list=os.listdir(os.path.join(train_path,'train',type_list[type_idx]))
        folder_idx=int(np.random.randint(0,len(folder_list),1))
        train_folder=os.path.join(train_path,'train',type_list[type_idx],folder_list[folder_idx])
        gm.set_value("train_folder",train_folder)
        for epoch in range(1, args.epochs + 1):
            train_loss = 0.
            train_error = 0.
            count=0
            t1=time.time()
            with tqdm(total=iteration) as t: 
                for batch_idx, (data, label) in enumerate(train_loader):
                    bag_label = label
                    # reset gradients
                    optimizer.zero_grad()
                    # label = label.cuda()
                    label = torch.tensor([type_idx]).cuda()
                    # # calculate loss and metrics
                    if max(label)==1:
                        bag_in_label=torch.tensor([1]).long()
                    elif max(label)==2:
                        bag_in_label=torch.tensor([2]).long()
                    elif max(label)==3:
                        bag_in_label=torch.tensor([3]).long()
                    elif max(label)==4:
                        bag_in_label=torch.tensor([4]).long()
                    else:
                        bag_in_label=torch.tensor([0]).long()
                    if args.cuda:
                        data, bag_label = data.cuda(), bag_label.cuda()
                    bag_in_label=bag_in_label.cuda()
                    data, bag_in_label = Variable(data), Variable(bag_in_label)                    
                    output_sum=model(data)
                    output=output_sum[0]
        
                    type_idx=int(np.random.randint(0,len(type_list),1))
                    folder_list=os.listdir(os.path.join(train_path,'train',type_list[type_idx]))
                    folder_idx=int(np.random.randint(0,len(folder_list),1))
                    train_folder=os.path.join(train_path,'train',type_list[type_idx],folder_list[folder_idx])
                    gm.set_value("train_folder",train_folder)
                    target=label.cuda()
        
                    loss=train_loss_fn(output,bag_in_label)
                    train_loss += loss.data
                    correct = output.argmax().eq(target.view_as(output.argmax())).sum().item()
                    train_error+=1-correct
                    loss.backward()
                    optimizer.step()
                    t.set_description('Processing epoch: '+str(epoch)+' train loss: '+str(train_loss/count))
                    t.update(1)
                    count+=1
                    if count>iteration:
                        training_time[epoch-1]=time.time()-t1
                        break
            scheduler.step()
            train_loss /= (count+1)
            train_error /= (count+1)
            training_loss[epoch-1]=train_loss # record the training loss
            
 #% 
    # if epoch%10==0:
        test_loader = data_utils.DataLoader(testset(),
                                             batch_size=256,
                                             shuffle=True,
                                             **loader_kwargs) 
        pred_type='test'
        folder_list=os.listdir(os.path.join(train_path,pred_type))
        fn=0
        test_loss=0
        bag_label=-1
        tp=0
        res_sum=np.zeros([0,6])
        c=0
        cm=np.zeros([5,5])
        with torch.no_grad():
            for folder in folder_list:
                bag_label+=1
                test_list=os.listdir(os.path.join(train_path,pred_type,folder))
                fn+=len(test_list)
                for file in test_list:
                    test_folder=os.path.join(train_path,pred_type,folder,file)
                    arr = np.arange(len(os.listdir(test_folder)))
                    np.random.shuffle(arr)
                    arr_idx=arr[:bag_length]            
                    gm.set_value("train_folder",test_folder)
                    gm.set_value("arr_idx",arr_idx)
                    count=0
                    
                    for batch_idx, (data, label) in enumerate(test_loader):
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
                            data,target = data.cuda(),target.cuda()
                        data = Variable(data)                       
                        output_sum=model(data)
                        output=output_sum[0]
                        loss=train_loss_fn(output,target)
                        test_loss += loss.data
                        pred_res=np.argmax(output.cpu().numpy()[0])
                        label_output=np.append([[bag_label]],output.cpu().numpy(),1)
                        res_sum=np.append(res_sum,label_output,0)
                        c+=1
                        cm[target.cpu().numpy()[0],pred_res]+=1
                        if pred_res==target.cpu().numpy()[0]:
                            tp+=1  
                            
                        break
        test_loss=test_loss/fn
        # print('test_loss: '+str(test_loss))
        # print('acc: '+str(tp/fn))
        # print(cm)
        testing_loss[epoch-1]=test_loss # record the training loss
        torch.save(model, 'D:/data/bone_marrow/mil_output/models/k_'+str(k)+'_'+str(bag_length)+'.pth') # 
        scio.savemat('D:/data/bone_marrow/mil_output/models/k_'+str(k)+'_'+str(bag_length)+'.mat', {'res_sum':res_sum})