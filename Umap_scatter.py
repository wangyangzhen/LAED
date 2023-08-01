# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:10:44 2023

@author: HP
"""

#%% scatter attention-feature
import cv2
import matplotlib.pyplot as plt
from MIL_model import resnext101_32x4d
c=0
patient_list=[]
repeat=10
dic_fn={}
dic_fn['0']=[]
dic_fn['1']=[]
dic_fn['2']=[]
dic_fn['3']=[]
dic_fn['4']=[]
fc_M=np.empty([0,512])
fc_C=np.empty([0,512])
fc_L=np.empty([0])
rep_record=np.zeros([repeat,1])
for k in range(1,2):
    model = torch.load('H:/data/bone_marrow/mil_output/roc_score/res_mat_618/external/k_indep_13_5.pth')
    # model=resnext101_32x8d(num_classes=21,pretrained=True)
    # model.fc = torch.nn.Linear(2048, 1024, bias=True)
    # if args.cuda:
    #     model.cuda()
    # state_dict=torch.load('net_params_1.pth')
    # model.load_state_dict(state_dict)
    train_path='H:/data/bone_marrow/k_bag_618/same_size/1_3/'
    f_output='H:/data/bone_marrow/mil_output/'
    cm=np.zeros([5,5])
    
    test_loss = 0.
    test_error = 0.
    # folder_idx=0
    pred_type='test'
    cm=np.zeros([5,5])
    test_loader = data_utils.DataLoader(testset(),
                                         batch_size=64,
                                         shuffle=True,
                                         **loader_kwargs) 
    folder_list=os.listdir(os.path.join(train_path,pred_type))
    test_list=[]
    acc=0
    fn=0
    bag_label=-1
    with torch.no_grad():
        for folder in folder_list:
            
            bag_label+=1
            
            test_list=os.listdir(os.path.join(train_path,pred_type,folder))
            fn+=len(test_list)
            for file in test_list:
                print(file)
                patient_list.append(file)
                dic_fn[str(bag_label)].append(file)
                test_folder=os.path.join(train_path,pred_type,folder,file)
                # print(test_folder)
                gm.set_value("train_folder",test_folder)
                count=0
                tp=0
                ouput_record=np.zeros([repeat,1])-1
                fc_L=np.append(fc_L,bag_label)
                for Rep_idx in range(repeat):
                    for batch_idx, (data, label) in enumerate(test_loader):
                        # print(len(label))
                        # bag_label = label
                        # if bag_label==1:
                        #     target=torch.tensor([1]).long()
                        # elif bag_label==2:
                        #     target=torch.tensor([2]).long()
                        # elif bag_label==3:
                        #     target=torch.tensor([3]).long()
                        # elif bag_label==4:
                        #     target=torch.tensor([4]).long()
                        # else:
                        #     target=torch.tensor([0]).long()
                        if args.cuda:
                            data = data.cuda()
                        data = Variable(data)
                        data = data.squeeze(0)
                        output_sum=model(data)
                        fc_M=np.append(fc_M,output_sum[5].cpu().numpy(),0)
                        if Rep_idx==0:
                            fc_C=np.append(fc_C,output_sum[6].cpu().numpy(),0)
                        break
#%%
import umap
import matplotlib.pyplot as plt
emb_U = umap.UMAP(n_neighbors=30,
                      n_components=2,
                      min_dist=0.3,
                      metric='correlation',
                      random_state=42).fit_transform(fc_C)
#%% scatter 4 hostipal data
train_path='H:/data/bone_marrow/k_bag_618/same_size/1_3/'
pred_type='test'
folder_list=os.listdir(os.path.join(train_path,pred_type))
t_n=[]
c=0
for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
    c=c+len(test_list)
    t_n.append(c)
sd=10
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
# plt.scatter(emb_U[:,0],emb_U[:,1],s=sd,color=[0.8,0.8,0.8])
all_idx=np.int32(np.linspace(0, t_n[0]*repeat*5-5, t_n[0]*repeat))
aml_idx=np.int32(np.linspace(t_n[0]*repeat*5+1, t_n[1]*repeat*5-4, (t_n[1]-t_n[0])*repeat))
apl_idx=np.int32(np.linspace(t_n[1]*repeat*5+2, t_n[2]*repeat*5-3, (t_n[2]-t_n[1])*repeat))
cll_idx=np.int32(np.linspace(t_n[2]*repeat*5+3, t_n[3]*repeat*5-2, (t_n[3]-t_n[2])*repeat))
cml_idx=np.int32(np.linspace(t_n[3]*repeat*5+4, t_n[4]*repeat*5-1, (t_n[4]-t_n[3])*repeat))
plt.scatter(emb_U[all_idx,0],emb_U[all_idx,1],s=sd,color=[0.83, 0.15, 0.15])
plt.scatter(emb_U[aml_idx,0],emb_U[aml_idx,1],s=sd,color=[0.14, 0.47, 0.71])
plt.scatter(emb_U[apl_idx,0],emb_U[apl_idx,1],s=sd,color=[0.12, 0.62, 0.4])
plt.scatter(emb_U[cll_idx,0],emb_U[cll_idx,1],s=sd,color=[0.94, 0.5, 0.14])
plt.scatter(emb_U[cml_idx,0],emb_U[cml_idx,1],s=sd,color=[0.52, 0.35, 0.64])

#%% scatter single-cell-feature
train_path='H:/data/bone_marrow/k_bag_618/same_size/1_3/'
folder_list=os.listdir(os.path.join(train_path,pred_type))
t_n=[]
c=0
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
    c=c+len(test_list)
    t_n.append(c)
for k in range(len(fc_L)):
    if fc_L[k]==0:
        plt.scatter(emb_U[k*256-256:k*256-128,0],emb_U[k*256-256:k*256-128,1],s=sd,color=[0.83, 0.15, 0.15])
    elif fc_L[k]==1:
        plt.scatter(emb_U[k*256-256:k*256-128,0],emb_U[k*256-256:k*256-128,1],s=sd,color=[0.14, 0.47, 0.71])
    elif fc_L[k]==2:
        plt.scatter(emb_U[k*256-256:k*256-128,0],emb_U[k*256-256:k*256-128,1],s=sd,color=[0.12, 0.62, 0.4])
    elif fc_L[k]==3:
        plt.scatter(emb_U[k*256-256:k*256-128,0],emb_U[k*256-256:k*256-128,1],s=sd,color=[0.94, 0.5, 0.14])
    else:
        plt.scatter(emb_U[k*256-256:k*256-128,0],emb_U[k*256-256:k*256-128,1],s=sd,color=[0.52, 0.35, 0.64])
