'''
Date: 2023-07-17 09:31:02
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-17 10:02:38
FilePath: /date/TMP_workshop/decouple_conv/ops_decouple.py
'''
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/date/TMP_workshop')
from decouple_conv.ops import *



class DAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True,):
        super(DAConv, self).__init__()
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.inp_dim=in_channels
        self.out_dim=out_channels
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 

        self.cac = CAUnit(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size
                              ,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.cdc = DAUnit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
                              , stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.register_parameter('cac_theta', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('cdc_theta', nn.Parameter(torch.tensor(1.0)))
        print("Using {} \n  ".format(self.__class__.__name__))

    def forward(self,x):
        if self.training:
            return torch.sigmoid(self.cac_theta)*self.cac(x) \
                   + torch.sigmoid(self.cdc_theta)*self.cdc(x)
        else:
            return self.test_forward(x)

    def re_para(self):
        k = self.cac.weight.data.sum( dim=[2, 3] )
        loc = int(self.cac.weight.size(3) /2)
        cac_k = torch.clone (self.cac.weight.data)
        cac_k[:,:,loc,loc] += k

        k = self.cdc.weight.data.sum(dim=[2, 3])
        loc = int(self.cdc.weight.size(3) / 2)
        cdc_k = torch.clone(self.cdc.weight.data)
        cdc_k[:, :, loc, loc] -= k

        self.K = torch.sigmoid(self.cac_theta)*cac_k + torch.sigmoid(self.cdc_theta)*cdc_k
        self.K = self.K.to(self.cac.weight.device)

        if self.cac.bias  is not None:
            self.B = torch.sigmoid(self.cac_theta)*self.cac.bias.data \
                     + torch.sigmoid(self.cdc_theta)*self.cdc.bias.data
            self.B = self.B.to(self.cac.weight.device)
        else:
            self.B = None

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x , weight=self.K,bias=self.B,padding=self.padding,stride=self.stride)



_decouple_conv_dict = {
    'DAConv': DAConv,
}

def decouple_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1,groups=1,
                        bias = True,decouple_conv_name='DAConv'):
    if kernel_size==1 or padding == 0 :
        print('kernel_size==1 , cannot decouple this conv , transfer to conv')
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias =bias,dilation=dilation,groups=groups)
    else:
        return _decouple_conv_dict[decouple_conv_name](in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias =bias,dilation=dilation,groups=groups)

if __name__ == '__main__':
    a = torch.rand(size=(1,3,256,256))
    net = decouple_conv_layer(3,8,3,1,1,decouple_conv_name='DAConv')
    train_a = net(a)
    print(net)
    print(net(a).shape)
    net.eval()
    test_a = net(a)
    print(torch.mean(train_a-test_a))