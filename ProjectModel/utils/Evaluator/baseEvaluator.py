# -*- coding: UTF-8 -*-
"""
*@ description: 计算语义分割的评价指标
*@ name:	baseEvaluator.py
*@ author: dengbozfan
*@ time:	2025/04/16 15:34
"""

from ... import torch,nn

class Accumulator:
    """记录相关数据"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SegmentationMetric:

    def __init__(self, num_classes, device):

        self.num_classes = num_classes
        self.device = device
        self.confusionMatrix = torch.zeros((self.num_classes, self.num_classes)).to(device)
    
    def getConfusionMatrix(self, imgPredict, imgLabel):
        """
        description: 通FCN的score.py的fast_hist()函数, 计算混淆矩阵
            Args: 
                imgPredict : 预测值
                imgLabel : 真实值
                ignore_labels : 需要忽略的类标签
            return: 混淆矩阵
        """
        # 去除 [0,255] 之外的
        mask = (imgLabel >= 0) & (imgLabel < self.num_classes)

        # 获取唯一索引值
        label = self.num_classes * imgLabel[mask] + imgPredict[mask]
        # 统计数目
        count = torch.bincount(label, minlength=self.num_classes ** 2)
        # 变形为混淆矩阵
        confousionMatrix = count.view(self.num_classes, self.num_classes)

        return confousionMatrix.to(self.device)
    
    def addBatch(self,imgPredict, imgLabel):

        assert imgPredict.shape == imgLabel.shape, f"imgPredict.shape:{imgPredict.shape}, imgLabel.shape{imgLabel.shape}"

        add_metrix = self.getConfusionMatrix(imgPredict,imgLabel)
        self.confusionMatrix += add_metrix

        return self.confusionMatrix
    
    def pixelAccuracy(self):
        """计算总像素的准确率 正确像素占总像素的比例"""
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    
    def classPixelAccuracy(self):
        """计算每个类的像素的准确率"""
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
    
    def meanPixelAccuracy(self):
        """计算类平均像素准确率"""
        claccAcc = self.classPixelAccuracy()
        meanAcc = claccAcc[claccAcc < float('inf')].mean()
        return meanAcc
    
    def IntersectionOverUnion(self):
        """计算交并比"""
        # 计算对角线上的值
        intersection = torch.diag(self.confusionMatrix)
        # 计算总和
        union = torch.sum(self.confusionMatrix, axis=1) + \
                torch.sum(self.confusionMatrix, axis=0) - \
                torch.diag(self.confusionMatrix)
        # 计算交并比
        Iou = intersection / union
        return Iou
    
    def meanIntersectionOverUnion(self):
        """计算平均交并比"""
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean() # 求各类别IoU的平均
        return mIoU
    
    def frequency_weighted_iou(self) -> float:
        """计算频率加权IoU"""
        intersection = torch.diag(self.confusionMatrix)
        union = self.confusionMatrix.sum(dim=0) + self.confusionMatrix.sum(dim=1) - intersection
        freq = self.confusionMatrix.sum(dim=1) / self.confusionMatrix.sum()
        return (freq * (intersection / union.where(union != 0, torch.tensor(1.0, device=self.device)))).sum().item()
    
    def reset(self):
        self.confusionMatrix = torch.zeros((self.num_classes, self.num_classes)).to(self.device)
    
def evaluate_segmentation_loss(net, data_iter, loss_func, device=None):
    """计算语义分割的平均损失"""

    if isinstance(net, nn.Module):
        net.eval()  # 设置模型为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 损失总和，样本数
    metric = Accumulator(2)
    with torch.no_grad():
        for image, mask in data_iter:
            image,mask = image.to(device), mask.to(device)
            y_hat = net(image)
            loss = loss_func(y_hat, mask)
            metric.add(float(loss.mean()), 1)
    return metric[0] / metric[1]

def evaluate_segmentation_iou(net, data_iter, num_classes, device=None):
    """计算语义分割的平均IoU"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置模型为评估模式
        if not device:
            device = next(iter(net.parameters())).device

    metric_log = Accumulator(2)
    metric = SegmentationMetric(num_classes,device)

    with torch.no_grad():
        for image, mask in data_iter:
            image,mask = image.to(device), mask.to(device)
            y_hat = net(image)
            y_hat_class = torch.argmax(y_hat, dim=1)
            metric.addBatch(y_hat_class, mask)
            mean_iou = metric.meanIntersectionOverUnion()
            metric_log.add(mean_iou, 1)  # 每个batch计算一次IoU

    return metric_log[0] / metric_log[1]

