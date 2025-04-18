# -*- coding: UTF-8 -*-
"""
*@ description: 绘制结果
*@ name:	plot_result.py
*@ author: dengbozfan
*@ time:	2025/04/17 18:28
"""
from ... import os,csv, plt, torch,np,ic
from .baseEvaluator import SegmentationMetric

class PlotResult:

    def __init__(self,
                 train_log_path:str,
                 save_dir:str):
        """
        绘制语义分割训练曲线类
        Args:
            train_log_path (str): 训练数据存储路径
            save_dir (str): 存储绘制图片的输出目录
        """
        self.train_log_path = train_log_path
        self.save_dir = save_dir
        
    def get_data_form_csv(self, trainLog_csv_path:str):
        """从CSV文件中读取数据
        [Epoch, Train Loss, Train Acc, Val Loss, Test Acc, LR]
        """
        datas = []
        with open(trainLog_csv_path, mode='r', encoding='utf-8') as file:
            lines = file.readlines()[1:]  # 跳过第一行
            for line in lines:
                row = line.strip().split(',')
                epoch, train_loss, train_acc, val_loss, test_acc, lr = row
                data = epoch, float(train_loss), float(train_acc), float(val_loss), float(test_acc), float(lr)
                datas.append(data)
        return zip(*datas)
    

    def plot_loss(self, trainLog_csv_path, save_dir=None):
        """绘制损失曲线
        
        """
        epochs, train_losses, _, val_losses, _, _ = self.get_data_form_csv(trainLog_csv_path)
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'train_val_loss.png'))
        plt.close()
    
    def plot_acc(self, trainLog_csv_path, save_dir=None):
        """绘制准确率曲线"""
        epochs, _, train_accs, _, test_accs, _ = self.get_data_form_csv(trainLog_csv_path)
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, train_accs, label='Training Accuracy')
        plt.plot(epochs, test_accs, label='Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'train_test_acc.png'))
        plt.close()

    def plot_current_lr(self, trainLog_csv_path, save_dir=None):
        """绘制学习率变化曲线"""
        epochs, _, _, _, _, lrs = self.get_data_form_csv(trainLog_csv_path)
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, lrs, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'lr.png'))
        plt.close()
    
    def compute_and_plot_confusion_matrix(self,epoch: int, num_classes, model, test_loader, device):
        """计算并绘制混淆矩阵"""
        # 在测试集上运行模型，获取 imgPredict 和 imgLabel
        imgPredict_list = []
        imgLabel_list = []

        model.eval()
        with torch.no_grad():
            for image, mask in test_loader:
                image, mask = image.to(device), mask.to(device)
                y_hat = model(image)
                y_hat_class = torch.argmax(y_hat, dim=1)  # 获取预测类别
                imgPredict_list.append(y_hat_class.cpu())
                imgLabel_list.append(mask.cpu())

        # 合并所有批次的数据
        imgPredict = torch.cat(imgPredict_list, dim=0)
        imgLabel = torch.cat(imgLabel_list, dim=0)

        # 计算混淆矩阵
        metric = SegmentationMetric(num_classes, device=device)
        metric.addBatch(imgPredict, imgLabel)
        confusion_matrix = metric.confusionMatrix.cpu().numpy()

        # 绘制混淆矩阵
        self.plot_confusion_matrix(confusion_matrix, epoch)

        return confusion_matrix
    

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, epoch: int):
        """绘制混淆矩阵的热图，并添加归一化操作和优化显示效果"""

        # 归一化操作
        normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        normalized_confusion_matrix = np.nan_to_num(normalized_confusion_matrix)  # 处理可能出现的除零情况

        # 绘制热图
        plt.figure(figsize=(10, 8))
        plt.imshow(normalized_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Normalized Confusion Matrix - Epoch {epoch}', fontsize=16)
        plt.colorbar()

        # 添加坐标轴标签
        tick_marks = np.arange(len(confusion_matrix))
        plt.xticks(tick_marks, tick_marks, fontsize=12)
        plt.yticks(tick_marks, tick_marks, fontsize=12)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)

        # 添加数值（显示归一化后的值，保留两位小数）
        for i in range(len(normalized_confusion_matrix)):
            for j in range(len(normalized_confusion_matrix[0])):
                plt.text(j, i, f"{normalized_confusion_matrix[i, j]:.2f}", 
                        ha='center', va='center', 
                        color='white' if normalized_confusion_matrix[i, j] > 0.5 else 'black',
                        fontsize=24)
        
        # 优化图像布局
        plt.tight_layout()

        # 保存混淆矩阵图像
        confusion_matrix_path = os.path.join(self.save_dir, f"confusion_matrix_epoch{epoch}.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plotAll(self):
        """绘制曲线: loss acc lr"""
        self.plot_loss(self.train_log_path,self.save_dir)
        self.plot_acc(self.train_log_path,self.save_dir)
        self.plot_current_lr(self.train_log_path,self.save_dir)
        