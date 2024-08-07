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


from torch.autograd import Variable
import torch.nn.functional as F
class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()

        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            # nn.Conv2d(6, 32, 3, 1, 1),
             nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
             nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
             nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
             nn.Conv2d( 32,32, 3, 1, 1),
            nn.ReLU(),
             nn.Conv2d( 32,32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
             nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
             nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
             nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
             nn.Conv2d( 32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
             nn.Conv2d( 32, 32, 3, 1, 1),
            nn.ReLU(),
             nn.Conv2d( 32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
             nn.Conv2d( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
             nn.Conv2d( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
             nn.Conv2d( 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
             nn.Conv2d( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
             nn.Conv2d( 32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list

class PReNet_decouple(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_decouple, self).__init__()

        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            decouple_conv_layer(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            decouple_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU(),
            decouple_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            decouple_conv_layer( 32,32, 3, 1, 1),
            nn.ReLU(),
            decouple_conv_layer( 32,32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            decouple_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU(),
            decouple_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            decouple_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU(),
            decouple_conv_layer( 32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            decouple_conv_layer( 32, 32, 3, 1, 1),
            nn.ReLU(),
            decouple_conv_layer( 32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            decouple_conv_layer( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            decouple_conv_layer( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            decouple_conv_layer( 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            decouple_conv_layer( 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            decouple_conv_layer( 32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


    
if __name__ == "__main__":
    import copy
    net = PReNet_decouple().cuda()
    copy_net = copy.deepcopy(net)
    
    a = torch.randn(size=(1,3,64,64)).cuda()
    a_train = net(a)[0]
    net.eval()
    a_test = net(a)[0]

    net_merge = merge_network_re(net)
    a_merge = net_merge(a)[0]

    print("\n \n \n \n \n original model",copy_net)
    print("\n \n \n \n \n merged model",net_merge)
    
    print("The average decouple_model error of train and test :",torch.mean(torch.abs(a_train-a_test)))
    print("The average decouple_model error of original and merge :",torch.mean(torch.abs(a_merge-a_test)))


    net_conv = PReNet()
    net_conv.eval()

    
    torch.save(copy_net,'/home/date/TMP_workshop/net_decouple.pth')
    torch.save(net_merge,'/home/date/TMP_workshop/net_merge.pth')
    torch.save(net_conv,'/home/date/TMP_workshop/net_conv.pth')