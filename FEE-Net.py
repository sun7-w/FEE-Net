import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['EFESNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
from modules.deablock import *
from modules.cga import *
from modules.attention import *
from modules.attention import BiLevelRoutingAttention_nchw



class PFC(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class EFESNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=224, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super(EFESNet, self).__init__()
        base_dim = 16
        
        self.pfc1 = PFC(in_channels=input_channels, out_channels=base_dim)
        self.fe_level_1 = nn.Conv2d(in_channels=base_dim, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.deablock_en1_1 = DEBlockTrain(default_conv, 32, 3)
        self.deablock_en1_2 = DEBlockTrain(default_conv, 32, 3)
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.deablock_en2_1 = DEBlockTrain(default_conv, 64, 3)
        self.deablock_en2_2 = DEBlockTrain(default_conv, 64, 3)
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 8, kernel_size=3, stride=1, padding=1)
        self.deablock_en3_1 = DEBlockTrain(default_conv, 128, 3)
        self.deablock_en3_2 = DEBlockTrain(default_conv, 128, 3)
        self.fe_level_4 = nn.Conv2d(in_channels=base_dim * 8, out_channels=base_dim * 16, kernel_size=3, stride=1, padding=1)
        self.deablock_en4_1 = DEBlockTrain(default_conv, 256, 3)
        self.deablock_en4_2 = DEBlockTrain(default_conv, 256, 3)

        #bottleneck
        self.deablock_bottleneck1 = DEBlockTrain(default_conv, 256, 3)
        self.deablock_bottleneck2 = BiLevelRoutingAttention_nchw(256)
        self.deablock_bottleneck3 = BiLevelRoutingAttention_nchw(256)
        self.deablock_bottleneck4 = DEBlockTrain(default_conv, 256, 3)

        
        #skip connection
        self.fusion_1 = Fusion(base_dim * 16,base_dim * 16)
        self.fusion_2 = Fusion(base_dim * 8,base_dim * 8)
        self.fusion_3 = Fusion(base_dim * 4,base_dim * 4)
        self.fusion_4 = Fusion(base_dim * 2,base_dim * 2)


        self.up_1 = nn.Sequential(nn.ConvTranspose2d(base_dim*16, base_dim*8, kernel_size=3, stride=2, padding=2, output_padding=1),
                                 nn.ReLU(True))
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(base_dim*8, base_dim*4, kernel_size=3, stride=2, padding=2, output_padding=1),
                                 nn.ReLU(True))
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=2, output_padding=1),
                                 nn.ReLU(True))
        self.up_4 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=2, output_padding=1),
                                 nn.ReLU(True))
        self.up_5 = nn.Conv2d(base_dim, num_classes, kernel_size=3, stride=1, padding=1)
        
        self.deablock_de1_1 = DEBlockTrain(default_conv, 256, 3)
        self.deablock_de1_2 = DEBlockTrain(default_conv, 256, 3)
        self.deablock_de1_3 = DEBlockTrain(default_conv, 256, 3)
        self.deablock_de2_1 = DEBlockTrain(default_conv, 128, 3)
        self.deablock_de2_2 = DEBlockTrain(default_conv, 128, 3)
        self.deablock_de2_3 = DEBlockTrain(default_conv, 128, 3)
        self.deablock_de3_1 = DEBlockTrain(default_conv, 64, 3)
        self.deablock_de3_2 = DEBlockTrain(default_conv, 64, 3)
        self.deablock_de3_3 = DEBlockTrain(default_conv, 64, 3)
        self.deablock_de4_1 = DEBlockTrain(default_conv, 32, 3)
        self.deablock_de4_2 = DEBlockTrain(default_conv, 32, 3)
        self.deablock_de4_3 = DEBlockTrain(default_conv, 32, 3)
        self.deablock_de5_1 = DEBlockTrain(default_conv, 16, 3)
        self.deablock_de5_2 = DEBlockTrain(default_conv, 16, 3)
        self.deablock_de5_3 = DEBlockTrain(default_conv, 16, 3)

        self.decoder = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.relu(F.max_pool2d(self.pfc1(x), kernel_size=2, stride=2))
        t1 = out
    
        ### Stage 2
        out = self.fe_level_1(out)
        out = self.deablock_en1_1(out)
        out = F.relu(F.max_pool2d(self.deablock_en1_2(out), kernel_size=2, stride=2))
        t2 = out
    
        ### Stage 3
        out = self.fe_level_2(out)
        out=self.deablock_en2_1(out)
        out = F.relu(F.max_pool2d(self.deablock_en2_2(out), kernel_size=2, stride=2))
        t3 = out
    
        ### Stage 4
        out = self.fe_level_3(out)
        out = self.deablock_en3_1(out)
        out = F.relu(F.max_pool2d(self.deablock_en3_2(out), kernel_size=2, stride=2))
        t4 = out

        ### Stage 5
        out = self.fe_level_4(out)
        out = self.deablock_en4_1(out)
        out = F.relu(F.max_pool2d(self.deablock_en4_2(out), kernel_size=2, stride=2))
        t5 = out
    
        ### Bottleneck
        t6 = self.deablock_bottleneck1(out)
        t6 = self.deablock_bottleneck2(t6)
        t6 = self.deablock_bottleneck3(t6)
        t6 = self.deablock_bottleneck4(t6) + out


        ### DEcoder
        ### Stage 5
        out = self.deablock_de1_1(t6)
        out = self.deablock_de1_2(out)
        out = F.interpolate(out, size=t5.size()[2:], mode='bilinear', align_corners=False)
        out = self.fusion_1(out,t5)
        out = self.deablock_de1_3(out)
        out = self.up_1(out)
        
        ### Stage 4
        out = self.deablock_de2_1(out)
        out = self.deablock_de2_2(out)
        out = F.interpolate(out, size=t4.size()[2:], mode='bilinear', align_corners=False)
        out = self.fusion_2(out,t4)
        out = self.deablock_de2_3(out)
        out = self.up_2(out)
    
        ### Stage 3
        out = self.deablock_de3_1(out)
        out = self.deablock_de3_2(out)
        out = F.interpolate(out, size=t3.size()[2:], mode='bilinear', align_corners=False)
        out = self.fusion_3(out,t3)
        out = self.deablock_de3_3(out)
        out = self.up_3(out)
    
        ### Stage 2
        out = self.deablock_de4_1(out)
        out = self.deablock_de4_2(out)
        out = F.interpolate(out, size=t2.size()[2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t2)
        out = self.deablock_de4_3(out)
        out = self.up_4(out)

        ### Stage 1
        out = self.deablock_de5_1(out)
        out = self.deablock_de5_2(out)
        out = F.interpolate(out, size=t1.size()[2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t1)
        out = self.deablock_de5_3(out)
        out = self.up_5(out)
    
        # Ensure final output size matches target size
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
    
        return out
