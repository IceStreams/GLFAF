# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:06:23 2022

@author: Zuoxibing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AFF(nn.Module):
    '''
    AFF
    '''
    def __init__(self, channels=64, r=1):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=[1,inter_channels,1,1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=[1,channels,1,1]),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x, residual = input
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class PSA_Channel_Module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out

class PSA_Spatial_Module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        return spatial_out

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor, device):
        super(GCNLayer, self).__init__()
        self.device = device
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential( nn.Linear(input_dim, output_dim))
        
        nodes_count = self.A.shape[0]
        
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(self.device)
        self.mask = torch.ceil( self.A*0.00001)
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))

        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output, A_hat

class SSConv(nn.Module):
    '''
    Depthwise Separable Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )

        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out) 
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class GLFAF(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, hidden_size:int, device, model='normal'):
        super(GLFAF, self).__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.device = device
        self.hidden_size = hidden_size
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        
        layers_count=2
        
        # SFE
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i), nn.Conv2d(self.channel, self.hidden_size, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.hidden_size),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        # DSC
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.CNN_Branch.add_module(' CNN_Branch'+str(i),SSConv(self.hidden_size, self.hidden_size,kernel_size=5))
            else:
                self.CNN_Branch.add_module(' CNN_Branch' + str(i), SSConv(self.hidden_size, 2*self.hidden_size, kernel_size=5))
        # DGC
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch'+str(i), GCNLayer(self.hidden_size, self.hidden_size, self.A, self.device))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(self.hidden_size, 2*self.hidden_size, self.A, self.device))
        # PSA
        self.Channel_Att_Module = nn.Sequential(PSA_Channel_Module(2*self.hidden_size))
        self.Spatial_Att_Module = nn.Sequential(PSA_Spatial_Module(2*self.hidden_size))
        # AFF
        self.AFF_module = nn.Sequential(AFF(2*self.hidden_size, 1))
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(2*self.hidden_size, self.class_count))
    
    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape

        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise
        
        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)
        hx = clean_x

        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))

        #PSA_Channel_Module+PSA_Spatial_Module
        CNN_result = self.Channel_Att_Module(CNN_result)
        CNN_result = self.Spatial_Att_Module(CNN_result)
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        H = superpixels_flatten
        for i in range(len(self.GCN_Branch)): H, self.A = self.GCN_Branch[i](H)
            
        H1 = H
        GCN_result = torch.matmul(self.Q, H1)

        #PSA_Channel_Module+PSA_Spatial_Module
        GCN_result = self.Channel_Att_Module(torch.unsqueeze(GCN_result.reshape(h, w, -1).permute([2, 0, 1]), 0))
        GCN_result = self.Spatial_Att_Module(GCN_result)
        GCN_result = torch.squeeze(GCN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        
        #AFF
        CNN_result = torch.unsqueeze(CNN_result.reshape(h, w, -1).permute([2, 0, 1]), 0)
        GCN_result = torch.unsqueeze(GCN_result.reshape(h, w, -1).permute([2, 0, 1]), 0)
        Y = self.AFF_module((CNN_result, GCN_result))      #
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])

        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y

