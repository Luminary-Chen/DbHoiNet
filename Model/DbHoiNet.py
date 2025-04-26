import torch
import torch.nn as nn
from timm.layers import  trunc_normal_ 

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, mod="star"):  
        super().__init__()
        self.mod = mod
        self.dwconv = ConvBN(dim, dim, (7, 1), 1, (3, 0), groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)  
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)  
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)    
        self.dwconv2 = ConvBN(dim, dim, (7, 1), 1, (3, 0), groups=dim, with_bn=False)
        self.act = nn.ReLU6()  

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2 if self.mod == "star" else  self.act(x1) + x2   
        x = self.dwconv2(self.g(x))
        x = input + x
        return x

class DbHoiNet(nn.Module):
    def __init__(self, base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3,  num_classes=6, **kwargs):  
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=(2,1), padding=1), nn.ReLU6()) 
        dpr = [x.item() for x in torch.linspace(0, sum(depths))] 
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1) 
            self.in_channel = embed_dim
            if i_layer > 4:  # all star
                blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i], mod="sum") for i in range(depths[i_layer])]   
            else:
                blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i], mod="star") for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights) 

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)










