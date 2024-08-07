'''
Date: 2023-07-17 09:31:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-17 10:03:23
FilePath: /date/TMP_workshop/decouple_conv/merge_network.py
'''
import torch
from torch import nn 
import sys 
sys.path.append('/home/date/TMP_workshop')
from decouple_conv.ops_decouple import decouple_conv_layer
import copy

def merge_network(model):
    model.eval()
    for name,i in model.named_children():
        if 'DAConv' in i.__class__.__name__:
            i.re_para()
            if i.B is not None:
                conv_net = nn.Conv2d(in_channels=i.inp_dim, out_channels=i.out_dim,kernel_size=i.kernel_size,bias=i.bias,stride=i.stride,padding=i.padding)
                conv_net.eval()
                conv_net.bias = nn.Parameter(i.B)
                conv_net.weight = nn.Parameter(i.K)
            else:
                conv_net = nn.Conv2d(in_channels=i.inp_dim, out_channels=i.out_dim,kernel_size=i.kernel_size,bias=False,stride=i.stride,padding=i.padding)
                conv_net.eval()
                conv_net.weight = nn.Parameter(i.K)
            model.__setattr__(name,conv_net)

    return model 


def merge_network_re(model):
    if model == None:
        return None
    
    for name,i in model.named_children():
        if 'DAConv' in i.__class__.__name__ :
            i.re_para()
            if i.B is not None:
                conv_net = nn.Conv2d(in_channels=i.inp_dim, out_channels=i.out_dim,kernel_size=i.kernel_size,bias=i.bias,stride=i.stride,padding=i.padding)
                conv_net.eval()
                conv_net.bias = nn.Parameter(i.B)
                conv_net.weight = nn.Parameter(i.K)
            else:
                conv_net = nn.Conv2d(in_channels=i.inp_dim, out_channels=i.out_dim,kernel_size=i.kernel_size,bias=False,stride=i.stride,padding=i.padding)
                conv_net.eval()
                conv_net.weight = nn.Parameter(i.K)
            model.__setattr__(name,conv_net)
        else:
            model.__setattr__(name,merge_network(i))
            merge_network_re(i)

    return model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np
from decouple_conv.ops_decouple import decouple_conv_layer



class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = decouple_conv_layer(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = decouple_conv_layer(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = decouple_conv_layer(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = decouple_conv_layer(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = decouple_conv_layer(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = decouple_conv_layer(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = decouple_conv_layer(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r




class enhance_net_nopool_CNN(nn.Module):

	def __init__(self):
		super(enhance_net_nopool_CNN, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r




if __name__ == "__main__":
	#---------------------------Merge------------
    net = enhance_net_nopool().cuda()
    weight = '/home/user/Desktop/date/Exposure-Workshop/Zero-DCE_code_DAConv/snapshots/Epoch1.pth'
    new_weight =  '/home/user/Desktop/date/Exposure-Workshop/Zero-DCE_code_DAConv/snapshots/Epoch1_merged.pth'
    net.load_state_dict(torch.load(weight))
    net_copy = copy.deepcopy(net)
    merge_network_re(net)
    print(net)
    torch.save(net.state_dict(),new_weight)
    a=torch.randn(size=(1,3,256,256)).cuda()
    a = torch.clamp(a,0,1)
    
    # net_before_merged = enhance_net_nopool().cuda()
    # net_before_merged.load_state_dict(torch.load(weight))
    
    # net_after_merged = enhance_net_nopool_CNN().cuda()
    # net_after_merged.load_state_dict(torch.load(new_weight))

    print(torch.mean(torch.abs(net_copy(a)[0] - net(a)[0])))
    