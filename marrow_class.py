# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:22:16 2021

@author: Administrator
"""

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from resnet_raw import resnext101_32x8d,resnet101
# from marrowdataloader import MarrowBags
from MarrowClasLoader import trainset,testset
from model_test import Attention, GatedAttention
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm 
import time
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=64, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=0, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=1, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=16, metavar='NTest',
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

train_loader = data_utils.DataLoader(trainset(),
                                     batch_size=args.mean_bag_length,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(testset(),
                                      batch_size=args.num_bags_test,
                                      shuffle=True,
                                      **loader_kwargs)
print('Init Model')
# if args.model=='attention':
#     model = Attention()
# elif args.model=='gated_attention':
#     model = GatedAttention()

#%%
# model = Attention()
# args.lr=0.00005
# args.reg=0.00001
model=resnext101_32x8d(num_classes=1000,pretrained=True)
model.fc = torch.nn.Linear(2048, 21, bias=True)
# for k,v in model.named_parameters():
#     if not('attention' in k or 'classifier' in k or 'fc' in k):
#         v.requires_grad=False#
# model.classifier = torch.nn.Sequential(torch.nn.Linear(512, 4096),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Dropout(p=0.5),
#                                         torch.nn.Linear(4096, 4096),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Dropout(p=0.5),
#                                         torch.nn.Linear(4096, 2))
if args.cuda:
    model.cuda()
#%%
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.2, last_epoch=-1)
writer = SummaryWriter('runs/exp')
graph_inputs=torch.from_numpy(np.random.rand(1,3,224,224)).type(torch.FloatTensor).cuda()
writer.add_graph(model, input_to_model=graph_inputs)
model.train()
train_loss = 0.
train_error = 0.
train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
fc=[]
for epoch in range(1, args.epochs + 1): 
    with tqdm(total=len(train_loader)) as t: 
        count=0
        for batch_idx, (data, label) in enumerate(train_loader):
            bag_label = label
            # reset gradients
            optimizer.zero_grad()
            # # calculate loss and metrics
            # if bag_label==9:
            #     bag_in_label=torch.tensor([1]).long()
            # else:
            #     bag_in_label=torch.tensor([0]).long()
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            target=label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            # data = data.squeeze(0)
            output=model(data)
            # fc.append(list(aa.cpu().detach().numpy()))
            # print('**************')
            # print(ab)
            # print(label[1])
            # print(asd)
            # print(bag_label)
            # loss, _ = model.calculate_objective(data, bag_label)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            loss=train_loss_fn(output,bag_label.long())
            # train_loss += loss.data[0]
            train_loss+=loss.item()
            train_error+=(len(label)-correct)/len(label)
            t.set_description('Iteration %i' % count)
            t.set_postfix(loss=loss.item(),error=(len(label)-correct)/len(label))

            count+=1
            t.update(1)
            # print(correct)
            # print(loss)
            # backward pass
            loss.backward()
            # step
            # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, lr: {:.7f}'.format(epoch, loss.item(), (len(label)-correct)/len(label), optimizer.state_dict()['param_groups'][0]['lr']))
            optimizer.step()
    t.close()
    scheduler.step()
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    writer.add_scalar('add_scalar/loss', train_loss, global_step=epoch)
    writer.add_scalar('add_scalar/error', train_error, global_step=epoch)
    # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, lr: {:.7f}'.format(epoch, train_loss, train_error, optimizer.state_dict()['param_groups'][0]['lr']))
    #%
    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label
            target=label.cuda()
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            output=model(data)
            data=torch.unsqueeze(data, dim=0)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct = pred.eq(target.view_as(pred)).sum().item()
            # print(loss)
            loss=train_loss_fn(output,bag_label.long())
            test_loss+=loss.item()
            test_error+=(len(label)-correct)/len(label)
    
        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        
        # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))
        print('Epoch: {}, Loss: {:.4f}, test error: {:.4f}, lr: {:.7f}'.format(epoch, test_loss, test_error, optimizer.state_dict()['param_groups'][0]['lr']))
torch.save(model.state_dict(), 'D:/program/AttentionDeepMIL-master/marrow_class_resnext50_20220405.pth')
