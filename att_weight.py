import cv2
import matplotlib.pyplot as plt
model = torch.load('D:/data/bone_marrow/mil_output/models/b64_k4.pth')
train_path='D:/data/bone_marrow/training/bag_class_k2'
f_output='D:/data/bone_marrow/mil_output/'
cm=np.zeros([5,5])
repeat=1
test_loss = 0.
test_error = 0.
# folder_idx=0
pred_type='test'
cm=np.zeros([5,5])
rep_record=np.zeros([repeat,1])
ouput_record=np.zeros([repeat,1])-1
folder_list=os.listdir(os.path.join(train_path,pred_type))
test_list=[]
acc=0
fn=0
bag_label=-1
test_loader = data_utils.DataLoader(testset(),
                                     batch_size=64,
                                     shuffle=True,
                                     **loader_kwargs)
att_weight_sum=[]
for k in range(1):
    with torch.no_grad():
        folder=folder_list[0]
        bag_label+=1
        test_list=os.listdir(os.path.join(train_path,pred_type,folder))[0]
        fn+=len(test_list)
        file=test_list
        test_folder=os.path.join(train_path,pred_type,folder,file)
        gm.set_value("train_folder",test_folder)
        count=0
        tp=0
        for batch_idx, (data, label) in enumerate(test_loader):
            # bag_label = label
            data = data.cuda()
            data = Variable(data)
            # data = data.squeeze(0)
            output_sum=model(data)
            output=output_sum[0]
            att_res=output_sum[3].cpu().numpy()
            M=output_sum[5].cpu().numpy()
            pred=output_sum[2][0].cpu().numpy()[0]
            att_weight=(output_sum[3].cpu().numpy())[pred,:]
            # att_weight=np.mean((output_sum[3].cpu().numpy()),0)
            att_weight_sum=np.append(att_weight_sum,att_weight)
            break
        test_image=gm.get_value('test_image')
        print(output_sum[0])
    
        for i in range(64):
            im=cv2.imread(test_image[i])
            im=(im-np.min(im))/(np.max(im)-np.min(im))*255;
            if att_weight[i]>=0:
                cv2.imwrite('D:/data/bone_marrow/mil_output/att_analysis/p'+str(att_weight[i])[0:5]+'.png',im)
            else:
                cv2.imwrite('D:/data/bone_marrow/mil_output/att_analysis/n'+str(att_weight[i])[0:5]+'.png',im)
#%%
import cv2
import matplotlib.pyplot as plt
c=0
patient_list=[]
repeat=20
dic_rec={}
dic_rec['0']=[]
dic_rec['1']=[]
dic_rec['2']=[]
dic_rec['3']=[]
dic_rec['4']=[]
dic_fc={}
dic_fc['0']=[]
dic_fc['1']=[]
dic_fc['2']=[]
dic_fc['3']=[]
dic_fc['4']=[]
dic_fn={}
dic_fn['0']=[]
dic_fn['1']=[]
dic_fn['2']=[]
dic_fn['3']=[]
dic_fn['4']=[]
rep_record=np.zeros([repeat,1])
for k in range(1,6):
    model = torch.load('D:/data/bone_marrow/mil_output/models/b64_k'+str(k)+'.pth')
    train_path='D:/data/bone_marrow/training/bag_class_k'+str(k)
    f_output='D:/data/bone_marrow/mil_output/'
    cm=np.zeros([5,5])
    
    test_loss = 0.
    test_error = 0.
    # folder_idx=0
    pred_type='test'
    cm=np.zeros([5,5])
    
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
                patient_list.append(file)
                dic_fn[str(bag_label)].append(file)
                test_folder=os.path.join(train_path,pred_type,folder,file)
                # print(test_folder)
                gm.set_value("train_folder",test_folder)
                count=0
                tp=0
                ouput_record=np.zeros([repeat,1])-1
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
                        data = data.cuda()
                    data = Variable(data)
                    # data = data.squeeze(0)
                    
                    output_sum=model(data)
                    if len(dic_fc[str(bag_label)])==0:
                        dic_fc[str(bag_label)]=output_sum[5].cpu().numpy()
                    else:
                        dic_fc[str(bag_label)]=np.append(dic_fc[str(bag_label)],output_sum[5].cpu().numpy(),0)
  
                    output=output_sum[0]
                    pred=output_sum[2][0].cpu().numpy()[0]
                    # ouput_record[count]=target
                    ouput_record[count,0]=pred
                    count+=1
                    # print(count)
                    ouput_record=np.uint8(ouput_record)    
                    counts = np.bincount(ouput_record[:,0])
                    pred_res=np.argmax(output.cpu().numpy()[0])
                    if pred_res==target.cpu().numpy()[0]:
                        tp+=1 
                    if batch_idx==repeat-1:                  
                        break
                if tp/repeat>0.5:
                    acc+=1
                
                ouput_record=ouput_record-bag_label
                rep_record=np.concatenate((rep_record,ouput_record),1)
                if len(dic_rec[str(bag_label)])==0:
                    dic_rec[str(bag_label)]=ouput_record
                else:
                    dic_rec[str(bag_label)]=np.append(dic_rec[str(bag_label)],ouput_record,1)
    print('acc: '+str(acc/fn))
#%%
import umap
import matplotlib.pyplot as plt
emb_U = umap.UMAP(n_neighbors=50,
                      n_components=2,
                      min_dist=0.8,
                      metric='correlation',
                      random_state=42).fit_transform(fc_M)
#%% scatter correct data
train_path='D:/data/bone_marrow/training/bag_class_k1'
pred_type='test'
folder_list=os.listdir(os.path.join(train_path,pred_type))
t_n=[]
c=0
for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
    c=c+len(test_list)
    t_n.append(c)
repeat=20
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.scatter(emb_U[:,0],emb_U[:,1],s=1,color=[0.8,0.8,0.8])
all_idx=np.int32(np.linspace(0, t_n[0]*repeat*5-5, t_n[0]*repeat))
aml_idx=np.int32(np.linspace(t_n[0]*repeat*5+1, t_n[1]*repeat*5-4, (t_n[1]-t_n[0])*repeat))
apl_idx=np.int32(np.linspace(t_n[1]*repeat*5+2, t_n[2]*repeat*5-3, (t_n[2]-t_n[1])*repeat))
cll_idx=np.int32(np.linspace(t_n[2]*repeat*5+3, t_n[3]*repeat*5-2, (t_n[3]-t_n[2])*repeat))
cml_idx=np.int32(np.linspace(t_n[3]*repeat*5+4, t_n[4]*repeat*5-1, (t_n[4]-t_n[3])*repeat))
plt.scatter(emb_U[all_idx,0],emb_U[all_idx,1],s=1,color=[0.83, 0.15, 0.15])
plt.scatter(emb_U[aml_idx,0],emb_U[aml_idx,1],s=1,color=[0.14, 0.47, 0.71])
plt.scatter(emb_U[apl_idx,0],emb_U[apl_idx,1],s=1,color=[0.12, 0.62, 0.4])
plt.scatter(emb_U[cll_idx,0],emb_U[cll_idx,1],s=1,color=[0.94, 0.5, 0.14])
plt.scatter(emb_U[cml_idx,0],emb_U[cml_idx,1],s=1,color=[0.52, 0.35, 0.64])
#%% scatter 4 hostipal data
train_path='D:/data/bone_marrow/training/bag_class_k1'
pred_type='test'
folder_list=os.listdir(os.path.join(train_path,pred_type))
t_n=[]
c=0
for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
    print(len(test_list))
    c=c+len(test_list)
    t_n.append(c)
repeat=100
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.scatter(emb_U[:,0],emb_U[:,1],s=1,color=[0.8,0.8,0.8])
aml_idx=np.int32(np.linspace(t_n[0]*repeat*5+1, t_n[1]*repeat*5-4, (t_n[1]-t_n[0])*repeat))
f_idx=[0,0,0,0,0,1,1,1,0,1,0,1,0,1,1,2,2,2,2,2,2,2,3,3]
for k in range(24):
    if f_idx[k]==0:
        plt.scatter(emb_U[aml_idx[k*100-100:k*100],0],emb_U[aml_idx[k*100-100:k*100],1],s=1,color=[0.83, 0.15, 0.15])
    elif f_idx[k]==1:
        plt.scatter(emb_U[aml_idx[k*100-100:k*100],0],emb_U[aml_idx[k*100-100:k*100],1],s=1,color=[0.14, 0.47, 0.71])
    elif f_idx[k]==2:
        plt.scatter(emb_U[aml_idx[k*100-100:k*100],0],emb_U[aml_idx[k*100-100:k*100],1],s=1,color=[0.12, 0.62, 0.4])
    elif f_idx[k]==3:
        plt.scatter(emb_U[aml_idx[k*100-100:k*100],0],emb_U[aml_idx[k*100-100:k*100],1],s=1,color=[0.94, 0.5, 0.14])
#%% scatter 4 hostipal data
train_path='D:/data/bone_marrow/training/bag_class_k1'
pred_type='test'
folder_list=os.listdir(os.path.join(train_path,pred_type))
t_n=[]
c=0
for folder in folder_list:
    bag_label+=1
    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
    print(len(test_list))
    c=c+len(test_list)
    t_n.append(c)
repeat=100
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.scatter(emb_U[:,0],emb_U[:,1],s=1,color=[0.8,0.8,0.8])
aml_idx=np.int32(np.linspace(t_n[0]*repeat*5+0, t_n[1]*repeat*5-5, (t_n[1]-t_n[0])*repeat*5))
f_idx=[0,0,0,0,0,1,1,1,0,1,0,1,0,1,1,2,2,2,2,2,2,2,3,3]
for k in range(24):
    if f_idx[k]==0:
        plt.scatter(emb_U[aml_idx[k*500-500:k*500],0],emb_U[aml_idx[k*500-500:k*500],1],s=1,color=[0.83, 0.15, 0.15])
    elif f_idx[k]==1:
        plt.scatter(emb_U[aml_idx[k*500-500:k*500],0],emb_U[aml_idx[k*500-500:k*500],1],s=1,color=[0.14, 0.47, 0.71])
    elif f_idx[k]==2:
        plt.scatter(emb_U[aml_idx[k*500-500:k*500],0],emb_U[aml_idx[k*500-500:k*500],1],s=1,color=[0.12, 0.62, 0.4])
    elif f_idx[k]==3:
        plt.scatter(emb_U[aml_idx[k*500-500:k*500],0],emb_U[aml_idx[k*500-500:k*500],1],s=1,color=[0.94, 0.5, 0.14])
# aml_idx=np.int32(np.linspace(t_n[0]*repeat*5+1, t_n[1]*repeat*5-4, (t_n[1]-t_n[0])*repeat))
# plt.scatter(emb_U[aml_idx,0],emb_U[aml_idx,1],s=1,color=[0.8, 0.8, 0.8])

#%%
import pickle
filename = 'D:/data/bone_marrow/mil_output/.data'
f = open(filename, 'wb')
# 将变量存储到目标文件中区
# pickle.dump({'fc_M':fc_M,'emb_U':emb_U}, f)
# pickle.dump({'dic_rec':dic_rec,'emb_U':emb_U,'dic_fc':dic_fc,'dic_fn':dic_fn}, f)
# 关闭文件
f.close() 
#%%
f=open('D:/data/bone_marrow/mil_output/umap_20220412.data',"rb")  
data=pickle.load(f)  
f.close()
fc_M=data['fc_M']
emb_U=data['emb_U']
#%%
f=open('D:/data/bone_marrow/mil_output/umap_5k_20220412.data',"rb")  
data=pickle.load(f)  
f.close()
dic_fc=data['dic_fc']
dic_fn=data['dic_fn']
dic_rec=data['dic_rec']
emb_U=data['emb_U']