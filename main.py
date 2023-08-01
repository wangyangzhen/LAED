# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 01:41:33 2022

@author: yangzhen
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
from models.model_laed import resnext101_32x4d
import GlobalManager as gm
gm._init_()
from marrowdataloader import trainset,testset
from tqdm import tqdm 
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# Training settings 
parser = argparse.ArgumentParser(description='Leukemia type prediction using deep learning')
parser.add_argument('--data_path', type=str, default='',
                    help='path to dataset')
parser.add_argument('--num_classes', type=int, default=5, metavar='N',
                    help='number of label classes')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                    help='learning rate (default: 0.00002)')
parser.add_argument('--reg', type=float, default=1e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--train_bag_length', type=int, default=128, metavar='ML',
                    help='average bag length')
parser.add_argument('--test_bag_length', type=int, default=1024, metavar='ML',
                    help='average bag length')
parser.add_argument('--backbone_checkpoint', type=str, default='',
                    help='average bag length')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train Set')
loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

print('Init Model')

data_path=args.data_path
type_list=os.listdir(os.path.join(data_path,'train'))
training_loss=np.zeros([args.epochs])
testing_loss=np.zeros([args.epochs])
training_time=np.zeros([args.epochs])
model=resnext101_32x4d(backbone_path=args.backbone_checkpoint,num_classes=args.num_classes)
train_loader = data_utils.DataLoader(trainset(),
                                     batch_size=args.train_bag_length,
                                     shuffle=True,
                                     **loader_kwargs)
model.fc = torch.nn.Linear(2048, 1024, bias=True)
if args.cuda:
    model.cuda()
iteration=0
for fn in os.listdir(os.path.join(data_path,'train')):
    iteration+=len(os.listdir(os.path.join(data_path,'train',fn)))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
fc=[]
type_idx=int(np.random.randint(0,len(type_list),1))
folder_list=os.listdir(os.path.join(data_path,'train',type_list[type_idx]))
folder_idx=int(np.random.randint(0,len(folder_list),1))
train_folder=os.path.join(data_path,'train',type_list[type_idx],folder_list[folder_idx])
gm.set_value("train_folder",train_folder)
for epoch in range(1, args.epochs + 1):
    train_loss = 0.
    train_error = 0.
    count=0
    t1=time.time()
    with tqdm(total=iteration) as t: 
        for batch_idx, (data, label) in enumerate(train_loader):
            
            optimizer.zero_grad()
            label = torch.tensor([type_idx]).cuda()
            # bag_label = label
            bag_label=torch.tensor([type_idx]).long()
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
           
            data, bag_label = Variable(data), Variable(bag_label)                    
            output_sum=model(data)
            output=output_sum[0]
            
            type_idx=int(np.random.randint(0,len(type_list),1))
            folder_list=os.listdir(os.path.join(data_path,'train',type_list[type_idx]))
            folder_idx=int(np.random.randint(0,len(folder_list),1))
            train_folder=os.path.join(data_path,'train',type_list[type_idx],folder_list[folder_idx])
            gm.set_value("train_folder",train_folder)
            
            loss=train_loss_fn(output,bag_label)
            train_loss += loss.data
            correct = output.argmax().eq(label.view_as(output.argmax())).sum().item()
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
       
#%%
print('Load Test Set')
pred_type='test'
folder_list=os.listdir(os.path.join(data_path,pred_type))
fn=0
test_loss=0
bag_label=-1
tp=0
res_sum=np.zeros([0,6])
c=0
cm=np.zeros([5,5])

for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(data_path,pred_type,folder))
    fn+=len(test_list)
    
    for file in test_list:
        print(file)
        test_folder=os.path.join(data_path,pred_type,folder,file)           
        gm.set_value("train_folder",test_folder)
        count=0
        test_loader = data_utils.DataLoader(testset(),
                                             batch_size=args.test_bag_length,
                                             shuffle=True,
                                             **loader_kwargs) 
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                # print(test_folder)
                print(len(label))
                target=torch.tensor([batch_idx]).long()
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
#%% Convert the true labels into binary format
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(res_sum[:,0])

# Calculate the AUC for each class 
auc_scores = []
for i in range(y_true_bin.shape[1]):
    auc_scores.append(roc_auc_score(y_true_bin[:, i], res_sum[:,i+1]))

# calculate macro-AUC
macro_auc = sum(auc_scores) / len(auc_scores)

print("AUC scores for each class:", auc_scores)
print("Macro-AUC:", macro_auc)
