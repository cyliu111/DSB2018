""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
from .parts import *
import torch.nn.functional as F
import cv2
import copy

def get_gaussian_kernel(size=5): # 获取高斯kerner 并转为tensor ，size 可以改变模糊程度
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def get_laplacian_kernel(n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.zeros(3,3, device = device)
    kernel[0,1], kernel[1,0], kernel[1,2], kernel[2,1] = 0.25, 0.25, 0.25, 0.25
    kernel[1,1] = -1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return  kernel.repeat(n_classes, 1, 1, 1)   

def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class AVEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AVEUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.kernel = get_gaussian_kernel(size=5)
      
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.Laplacian_kernel = get_laplacian_kernel(n_classes)
        # self.average = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)

    def average(self, x):
      return F.conv2d(x, self.Laplacian_kernel, padding=1, groups=self.n_classes)

    # def average(self, x):
    #     epsilon = 1e-6
    #     Dx = torch.diff(x, dim=2, append=x[0:,0:,0:1,0:])
    #     Dy = torch.diff(x, dim=3, append=x[0:,0:,0:,0:1])
    #     norm = torch.sqrt(torch.square(Dx) + torch.square(Dy) + epsilon)
    #     Dx_normed = torch.div(Dx,norm)
    #     Dy_normed = torch.div(Dy,norm)
    #     return torch.diff(Dx_normed, dim=2, prepend=Dx_normed[0:,0:,-1:,0:]) + torch.diff(Dy_normed, dim=3, prepend=Dy_normed[0:,0:,0:,-1:])

    # def average(self, x, nonezero_map):
    #     b = nonezero_map.sum(dim=(2,3), keepdim=True)
    #     a = (x*nonezero_map).sum(dim=(2,3), keepdim=True)
    #     c = torch.div(a,b)
    #     return x - c

    def forward(self, x, required_average):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = -self.logsoftmax(x)
        if required_average:
          for i in range(30): # 5 for enlarge 10 for noise
            x = x - 0.1*self.average(x)  
            # x[:,2:,:,:] = x[:,2:,:,:] + 0.05*self.average(x[:,2:,:,:], nonezero_map=1-(x.argmax(dim=1,keepdim=True)==2).float())                       
        return x   



