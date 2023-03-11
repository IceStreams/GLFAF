# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:06:23 2022

@author: Zuoxibing
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
from sklearn import preprocessing
import cv2
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import scipy.sparse as sp
import os

def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

def visualize_segmentation(img,segments,path):
    out = mark_boundaries(img[:,:,[0,1,2]], segments, color=(1, 1, 0), mode='thick')    #mode={‘thick’, ‘inner’, ‘outer’, ‘subpixel’}
    # out = (out[:, :, [0, 1, 2]]-np.min(out[:, :, [0, 1, 2]]))/(np.max(out[:, :, [0, 1, 2]])-np.min(out[:, :, [0, 1, 2]]))
    dpi = 300
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] * 5.0 / dpi, img.shape[0] * 5.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(out)
    fig.savefig(path+"/SLIC.eps", dpi=dpi)
    fig.savefig(path+"/SLIC.jpg", dpi=dpi)
    # plt.show()

class SLIC(object):
    def __init__(self, HSI, n_segments, compactness=0.01, max_iter=20, sigma=0, min_size_factor=0.3, max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        print("self.compactness:",self.compactness)
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 影像经过了一次LDA处理，需要再次归一化
        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        # data = (data - data.min()) * 1.0 / (data.max() - data.min())
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        
    
    def get_Q_and_S_and_Segments(self, result_path):
        img = self.data
        (h, w, d) = img.shape

        #计算超像素标签图
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,convert2lab=False,sigma=self.sigma, enforce_connectivity=True, min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False, start_label=0)
        
        # 判断超像素label是否连续,否则予以校正
        if segments.max()+1 != len(list(set(np.reshape(segments,[-1]).tolist()))): segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
        
        #显示超像素图片
        visualize_segmentation(img, segments, result_path)

        # 计算超像素S以及相关系数矩阵Q
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        
        self.S = S
        self.Q = Q
        return Q, S , self.segments

    def get_A(self, sigma: float):
        '''
         根据 segments 判定邻接矩阵
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss
        return A

class LDA_SLIC(object):
    def __init__(self, result_path, data, labels, n_component, compactness):
        self.result_path = result_path
        self.data=data
        self.init_labels=labels
        self.curr_data=data
        self.n_component=n_component
        self.compactness = compactness
        self.height,self.width,self.bands=data.shape
        self.x_flatt=np.reshape(data,[self.width*self.height, self.bands])
        self.y_flatt=np.reshape(labels,[self.height*self.width])
        self.labes=labels
        
    def LDA_Process(self,curr_labels):
        '''
        LDA降维最多降到类别数k-1的维数
        '''
        curr_labels=np.reshape(curr_labels,[-1])
        
        idx=np.where(curr_labels!=0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis()
        lda.fit(x,y-1)
        X_new = lda.transform(self.x_flatt)     
        return np.reshape(X_new,[self.height, self.width,-1])
       
    def SLIC_Process(self, img, scale):
        n_segments_init = self.height*self.width/scale
        print("n_segments_init",n_segments_init)
        myslic = SLIC(img, n_segments=n_segments_init, compactness = 1, sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments(self.result_path)                      
        #Q:超像素分配矩阵（像素索引，超像素索引），S:超像素数据集（超像素索引，超像素特征），Segments：超像素标签图，大小()
        
        A=myslic.get_A(sigma=10)
        return Q,S,A,Segments
        
    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q, S, A, Seg = self.SLIC_Process(X,scale=scale)
        return Q, S, A, Seg
    
    def simple_superpixel_no_LDA(self,scale):
        Q, S, A, Seg = self.SLIC_Process(self.data,scale=scale)
        return Q, S, A,Seg