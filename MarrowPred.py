# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:45:45 2022

@author: Administrator
"""
for f in range(2,6):
    model = torch.load('D:/data/bone_marrow/mil_output/models/b64_k'+str(f)+'.pth')
    train_path='D:/data/bone_marrow/training/bag_class_k'+str(f)
    f_output='D:/data/bone_marrow/mil_output/data_size/f'+str(f)+'/'
    os.mkdir(f_output)
    cm=np.zeros([5,5])
    repeat=11
    data_size_idx=[1,2,4,8,16,32,64,128,256,512]
    for k in range(len(data_size_idx)):
        data_size=data_size_idx[k]
        test_loss = 0.
        test_error = 0.
        # folder_idx=0
        pred_type='test'
        cm=np.zeros([5,5])
        
        ouput_record=np.zeros([repeat,1])-1
        folder_list=os.listdir(os.path.join(train_path,pred_type))
        acc_sum=[]
        rep_sum={}
        for kk in range(5):
            acc=0
            fn=0
            bag_label=-1
            test_list=[]
            rep_record=np.zeros([repeat,1])
            with torch.no_grad():
                for folder in folder_list:
                    bag_label+=1
                    test_list=os.listdir(os.path.join(train_path,pred_type,folder))
                    fn+=len(test_list)
                    for file in test_list:
                        test_folder=os.path.join(train_path,pred_type,folder,file)
                        arr = np.arange(len(os.listdir(test_folder)))
                        np.random.shuffle(arr)
                        arr_idx=arr[:data_size]            
                        gm.set_value("train_folder",test_folder)
                        gm.set_value("arr_idx",arr_idx)
                        count=0
                        tp=0
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
                            output=output_sum[0]
                            pred=output_sum[2][0].cpu().numpy()[0]
                            ouput_record[count,0]=pred
                            count+=1
                            # print(count)
                            ouput_record=np.uint8(ouput_record)    
                            counts = np.bincount(ouput_record[:,0])
                            pred_res=np.argmax(output.cpu().numpy()[0])
                            if pred_res==target.cpu().numpy()[0]:
                                tp+=1                   
                            if count>=repeat:
                                break
                        # print(test_folder)
                        # print(tp/repeat)
                        if tp/repeat>0.5:
                            acc+=1
                        # cm[bag_label,np.argmax(counts)]+=1
                        ouput_record=ouput_record-bag_label
                        rep_record=np.concatenate((rep_record,ouput_record),1)
            print('acc: '+str(acc/fn))
            acc_sum.append(acc/fn)
            rep_sum['k'+str(kk)]=rep_record
        scio.savemat(f_output+'bg_'+str(args.bag_length)+'_k_'+str(k)+'_20220415.mat', {'acc_sum':acc_sum,'rep_sum':rep_sum})