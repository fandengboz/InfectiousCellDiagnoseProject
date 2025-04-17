# -*- coding: UTF-8 -*-
"""
*@ description: 神经网络 U-net | 网络模型
*@ name:	model_unet.py
*@ author: dengbozfan and wangxuan 
*@ time:	2025/03/29 17:15
"""

from .. import torch, nn, ic
from .. import resnet18

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet_up_Resnet18(nn.Module):
    """使用Resnet18作为编码器的UNet网络"""

    def __init__(self, n_classes=4):
        super().__init__()
        self.n_classes = n_classes
        # 使用Resnet18作为编码器
        self.resnet = resnet18(pretrained=False)

        # 编码器部分
        self.pre = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            
        )  # 64x64, 64 通道

        self.enc1 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1 
         ) # 
        self.enc2 = self.resnet.layer2  # 32x32, 128 通道
        self.enc3 = self.resnet.layer3  # 16x16, 256 通道
        self.enc4 = self.resnet.layer4  # 8x8, 512 通道

        # 解码器部分
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 16x16
        self.dec4 = ConvBlock(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 32x32
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 64x64
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 128x128
        self.dec1 = ConvBlock(64 + 64, 64)

        # 最终输出
        self.invertpre = nn.ConvTranspose2d(64, 3, 2, stride=2)  # 256x256
        self.enddec = ConvBlock(3 + 3, 128)  # 输入是 invertpre 的输出和原始输入 
        self.final = nn.Conv2d(128, n_classes, 1)  # 输入通道数 128

    def forward(self, x):

        x_pre = self.pre(x)
        # ic(x.shape)
        # 编码器部分
        e1 = self.enc1(x_pre)
        # ic(e1.shape)
        e2 = self.enc2(e1)
        # ic(e2.shape)
        e3 = self.enc3(e2)
        # ic(e3.shape)
        e4 = self.enc4(e3)
        # ic(e4.shape)
        # 解码器部分
        # ic(self.up4(e4).shape)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        # ic(d4.shape)
        # ic(self.up3(d4).shape)
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        # ic(d3.shape)
        # ic(self.up2(d3).shape)
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        # ic(d2.shape)
        # ic(self.up1(d2).shape)
        d1 = self.dec1(torch.cat([self.up1(d2), x_pre], dim=1))
        # ic(d1.shape)
        # 最终输出
        end_dec = self.enddec(torch.cat([self.invertpre(d1),x],dim=1))
        out = self.final(end_dec)

        return torch.softmax(out, dim=1)