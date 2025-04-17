# -*- coding: UTF-8 -*-
"""
*@ description: 构建残差网络 | 用于图像分类
*@ name:	resnet.py
*@ author: dengbozfan
*@ time:	2025/03/22 10:21
"""

from .. import torch,nn,F



class BasicBlock(nn.Module):
    """基本残差模块"""
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.downsample(downsample)
        )
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.left(x)  # 这是由于残差块需要保留原始输入
        out += identity  # 这是ResNet的核心，在输出上叠加了输入x
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    
    # 每个Bottleneck块输出通道数相对于输入通道数的扩展倍数
    expansion = 4 
 
    # 这里相对于RseNet，在代码中增加一下两个参数groups和width_per_group（即为group数和conv2中组卷积每个group的卷积核个数）
    # 默认值就是正常的ResNet
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        # 这里也可以自动计算*中间的通道数*，也就是3x3卷积后的通道数，如果不改变就是out_channels
        # 如果groups=32,with_per_group=4,out_channels就翻倍了

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 1. 1*1卷积 降维
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # 2.接着进行组卷积, 返回组合后的特征图
        # 组卷积的数，需要传入参数
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # 3.1*1 卷积 升维
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):

        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out += identity  # 残差连接
        out = self.relu(out)
 
        return out
    
class ResNeXt(nn.Module):

    """搭建ResNeXt结构 选用 C 方式"""

    def __init__(self,
                 block,  # 表示 block 类
                 blocks_num,  # 表示的是每一层block的个数
                 num_classes=4,  # 表示类别
                 include_top=True,  # 表示是否含有分类层(可做迁移学习)
                 groups=1,  # 表示组卷积的分组数
                 width_per_group=64):
        
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
        # 1. 构建 输入
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])           # 64 -> 128
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)# 128 -> 256
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)# 256 -> 512
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2) # 512 -> 1024
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
 
    # 形成单个Stage的网络结构
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
            
        # 该部分是将每个blocks的第一个残差结构保存在layers列表中。
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion  # 得到最后的输出
 
        # 该部分是将每个blocks的剩下残差结构保存在layers列表中，这样就完成了一个blocks的构造。
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
         # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
        return nn.Sequential(*layers)
 
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
 
        return x


