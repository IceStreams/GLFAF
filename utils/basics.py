# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:06:23 2022

@author: Zuoxibing
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import h5py
import os
import time
#import pickle
import scipy.io as sio
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from sklearn import metrics

#获取HSI数据
def get_HSI(data_name):
    if data_name == "IN":
        img = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == "PU":
        img = sio.loadmat('./data/PaviaU.mat')['paviaU']
        gt = sio.loadmat('./data/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == "SA":
        img = sio.loadmat('./data/Salinas_corrected.mat')['salinas_corrected']
        gt = sio.loadmat('./data/Salinas_gt.mat')['salinas_gt']
    elif data_name == "LongKou":
        img = sio.loadmat('./data/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        gt = sio.loadmat('./data/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
    elif data_name == "HanChuan":
        img = sio.loadmat('./data/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        gt = sio.loadmat('./data/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
    elif data_name == "HongHu":
        img = sio.loadmat('./data/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        gt = sio.loadmat('./data/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    elif data_name == "Houston2013":
        img = sio.loadmat('./data/DFC2013_Houston.mat')['DFC2013_Houston']
        gt = sio.loadmat('./data/DFC2013_Houston_gt.mat')['DFC2013_Houston_gt']
    class_count = gt.max()
    return img, gt, class_count

def get_train_id_radomsample(label,n_from_each_class,random_seed):
    random.seed(random_seed)
    num_examples = len(label)
    label = np.array(label)
    class_num = label.max()
    raw_indices = np.arange(num_examples)
    i_labeled = []
    i_val = []
    i_test = []
    for c in range(class_num+1):
        if c ==0:
            continue
        if len(raw_indices[label == c]) > n_from_each_class:
            i = random.sample(list(raw_indices[label == c]), round(n_from_each_class*0.8))
            i_rest = set(list(raw_indices[label == c])) - set(i)
            j = random.sample(list(i_rest),  round(n_from_each_class*0.2))
        else:
            i = random.sample(list(raw_indices[label == c]), 16)      ##针对IN16中样本数目不足20的特殊情况
            i_rest = set(list(raw_indices[label == c])) - set(i_labeled)
            j = random.sample(list(i_rest), 4)
        i_labeled += list(i)
        i_val += list(j)
    # i_val = raw_indices
    i_test = raw_indices     #测试集不打乱顺序
    t_labels = list(label[i_labeled])
    from collections import Counter
    print("sample number perclass",Counter(t_labels))  # 验证是否每类地物选取n_from_each_class*0.8个
    return i_labeled, i_val, i_test

def colors_correspond_to_data(data_name, class_num):
    if data_name == "PU":
        colors = ['gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    elif data_name == "LongKou":
        colors = ['#ff0000', '#ee9a00', '#ffff00', '#00ff00', '#00ffff', '#008b8b', '#0000ff', '#ffffff', '#a020f0']
    elif data_name == "SA":
        colors = ['#f7afba', '#41a86c', '#f3a10f', '#4164ad', '#e41719', '#6b3c89', '#8b421e', '#b3c1c5', '#72c7d7', '#4f6530', '#ebe62e', '#8ec537', '#da429a', '#6db8b6', '#b37473', '#1f295b']
    else:
        colors = ['black','gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'blue', 'purple', 'red', 
        'yellow', 'steelblue', 'olive', 'sandybrown', 'lawngreen', 'darkorange', 'whitesmoke','tomato', 
        'lightsalmon', 'teal', 'olive', 'lightpink', 'gold', 'lightsteelblue'][:class_num]
    return colors

def classification_map_with_back(pred=None, ground_truth=None, dpi=600, save_path=None, data_name=None):
    class_num = ground_truth.max()
    class_map = pred.reshape(ground_truth.shape)
    colors = colors_correspond_to_data(data_name, class_num)
    cmap = mpl.colors.ListedColormap(colors)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 5.0 / dpi, ground_truth.shape[0] * 5.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    
    ax.imshow(class_map,cmap = cmap)
    fig.savefig(save_path+".eps", dpi=dpi)
    fig.savefig(save_path+".png", dpi=dpi)

def classification_map_without_back(pred=None, ground_truth=None, dpi=600, save_path=None):
    class_num = ground_truth.max()
    ground_truth_reshape = np.reshape(ground_truth, [-1])
    class_map = pred + 1
    print("class_map:",class_map.shape)
    print("ground_truth_reshape:",ground_truth_reshape.shape)
    class_map[ground_truth_reshape ==0] = 0
    class_map = class_map.reshape(ground_truth.shape)
    
    colors = ['black','gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'blue', 'purple', 'red', 
    'yellow', 'steelblue', 'olive', 'sandybrown', 'lawngreen', 'darkorange', 'whitesmoke','tomato', 
    'lightsalmon', 'teal', 'olive', 'lightpink', 'gold', 'lightsteelblue'][:class_num+1]
    cmap = mpl.colors.ListedColormap(colors)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 5.0 / dpi, ground_truth.shape[0] * 5.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    
    ax.imshow(class_map,cmap = cmap)
    fig.savefig(save_path+".eps", dpi=dpi)
    fig.savefig(save_path+".png", dpi=dpi)

def GT_To_One_Hot(gt, class_count):
    GT_One_Hot = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count,dtype=np.float32)
            if gt[i, j] != 0:
                temp[int( gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [gt.shape[0], gt.shape[1], class_count])
    return GT_One_Hot

def get_gt_onehot_mask(gt_reshape, height, width, class_count, train_data_index, val_data_index, test_data_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    Test_GT = np.reshape(test_samples_gt, [height, width])

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt,[height,width])
    test_samples_gt = np.reshape(test_samples_gt,[height,width])
    val_samples_gt = np.reshape(val_samples_gt,[height,width])

    train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt,class_count)
    test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt,class_count)
    val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt,class_count)

    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)
    test_samples_gt_onehot = np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)
    val_samples_gt_onehot = np.reshape(val_samples_gt_onehot,[-1,class_count]).astype(int)

    ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
    # 训练集
    train_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    train_samples_gt = np.reshape(train_samples_gt, [height * width])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_count])

    # 测试集
    test_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_count])

    # 验证集
    val_label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    val_samples_gt = np.reshape(val_samples_gt, [height * width])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_count])
    
    return train_samples_gt, test_samples_gt, val_samples_gt, train_samples_gt_onehot, test_samples_gt_onehot, val_samples_gt_onehot, train_label_mask, test_label_mask, val_label_mask, Test_GT

def isexists_dir_Create(paths):
     if not os.path.exists(paths):
        os.makedirs(paths)

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels,torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def compute_weighted_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    pixels_count = predict.shape[0]
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict + 1e-15))
    we = torch.mul(we, reallabel_mask)

    we2 = torch.sum(real_labels, 0)
    we2 = 1. / (we2 + 1)
    we2 = torch.unsqueeze(we2, 0)
    we2 = we2.repeat([pixels_count, 1])
    we = torch.mul(we, we2)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def train_eval_metrics(network_output,train_samples_gt,train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx=(train_samples_gt!=0).float()
        available_label_count=available_label_idx.sum()
        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
        OA= correct_prediction.cpu()/available_label_count
        return OA

def test_metrics(network_output,train_samples_gt,train_samples_gt_onehot, Test_GT, zeros, height, width, class_count):
    with torch.no_grad():
        #OA
        available_label_idx=(train_samples_gt!=0).float()
        available_label_count=available_label_idx.sum()
        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
        OA = correct_prediction.cpu()/available_label_count
        OA = OA.cpu().numpy()
        
        # AA
        zero_vector = np.zeros([class_count])
        output_data=network_output.cpu().numpy()
        train_samples_gt=train_samples_gt.cpu().numpy()
        train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
        
        output_data = np.reshape(output_data, [height * width, class_count])
        idx = np.argmax(output_data, axis=-1)
        for z in range(output_data.shape[0]):
            if ~(zero_vector == output_data[z]).all():
                idx[z] += 1
        # idx = idx + train_samples_gt
        count_perclass = np.zeros([class_count])
        correct_perclass = np.zeros([class_count])
        for x in range(len(train_samples_gt)):
            if train_samples_gt[x] != 0:
                count_perclass[int(train_samples_gt[x] - 1)] += 1
                if train_samples_gt[x] == idx[x]:
                    correct_perclass[int(train_samples_gt[x] - 1)] += 1
        test_AC_list = correct_perclass / count_perclass
        test_AA = np.average(test_AC_list)

        # Kappa
        test_pre_label_list = []
        test_real_label_list = []
        output_data = np.reshape(output_data, [height * width, class_count])
        idx = np.argmax(output_data, axis=-1)
        idx = np.reshape(idx, [height, width])
        for ii in range(height):
            for jj in range(width):
                if Test_GT[ii][jj] != 0:
                    test_pre_label_list.append(idx[ii][jj] + 1)
                    test_real_label_list.append(Test_GT[ii][jj])
        test_pre_label_list = np.array(test_pre_label_list)
        test_real_label_list = np.array(test_real_label_list)
        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                          test_real_label_list.astype(np.int16))
        test_kpp = kappa
        return OA, test_AA, test_kpp, test_AC_list
    