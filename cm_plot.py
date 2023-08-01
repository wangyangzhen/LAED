# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:29:39 2020

@author: yangzhen
"""
import itertools 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        cm = cm.astype('float') / np.repeat(np.expand_dims(cm.sum(axis=1), axis=1), 15, axis=1)
        store_items = pd.DataFrame(cm)
        store_items=store_items.fillna(0)
        cm=store_items.values
        print("Normalized confusion matrix")
        plt.figure(figsize=(16, 9))
        plot_norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0,clip=True)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys,norm=plot_norm)
    else:
        print('Confusion matrix, without normalization')
        plt.figure(figsize=(16, 9))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    	# 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    	# x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    	# 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        if np.isnan(thresh):
            thresh=0.5
        # num[np.isnan(num)]=0
        plt.text(j, i, num,
                  verticalalignment='center',
                  horizontalalignment="center",
                  color="white" if float(num) > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
