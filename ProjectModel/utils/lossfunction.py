# -*- coding: UTF-8 -*-
"""
*@ description: 定义损失函数
*@ name:	lossfunction.py
*@ author: dengbozfan
*@ time:	2025/04/18 17:40
"""

from .. import nn,F,torch

class DiceLoss(nn.Module):
    
    def __init__(self, num_classes, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        计算Dice Loss
        Args:
            preds (torch.Tensor): 模型输出(logits), 形状为 (batch_size, num_classes, height, width)
            targets (torch.Tensor): 标签，形状为 (batch_size, height, width)
        Returns:
            torch.Tensor: Dice Loss
        """
        # 对模型输出进行softmax处理
        preds = F.softmax(preds, dim=1)

        # 将标签转换为独热编码
        one_hot_targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 计算Dice系数
        intersection = torch.sum(preds * one_hot_targets, dim=(0, 2, 3))
        union = torch.sum(preds, dim=(0, 2, 3)) + torch.sum(one_hot_targets, dim=(0, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 返回Dice Loss
        return 1 - torch.mean(dice_score)
    