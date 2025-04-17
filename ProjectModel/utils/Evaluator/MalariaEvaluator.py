# -*- coding: UTF-8 -*-
"""
*@ description: 数据集评价函数
*@ name:	MalariaEvaluator.py
*@ author: dengbozfan
*@ time:	2025/04/17 19:19
"""

from .baseEvaluator import SegmentationMetric
from ..utils import to_serialize
from typing import Dict,Optional,List
from ... import np,json,torch,plt,nn,os,DataLoader,ic


class MalariaEvaluator:

    def __init__(self, num_classes:int,
                 model: nn.Module,
                 test_loader:DataLoader,
                 device,save_dir:Optional[str]):
        
        self.num_classes = num_classes
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.evaluator = SegmentationMetric(num_classes,device)
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir,exist_ok=True)

        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self):

        imgPredict_list = []
        imgLabel_list = []

        self.model.eval()
        with torch.no_grad():
            for image, mask in self.test_loader:
                image, mask = image.to(self.device), mask.to(self.device)
                y_hat = self.model(image)
                y_hat_class = torch.argmax(y_hat, dim=1)  # 获取预测类别
                imgPredict_list.append(y_hat_class.cpu())
                imgLabel_list.append(mask.cpu())

        # 合并所有批次的数据
        imgPredict = torch.cat(imgPredict_list, dim=0)
        imgLabel = torch.cat(imgLabel_list, dim=0)

        self.evaluator.addBatch(imgPredict, imgLabel)

    def get_results(self) -> Dict[str, float]:
        """获取所有评估指标"""
        return {
            "pixel_accuracy": self.evaluator.pixelAccuracy().cpu().numpy(),
            "mean_pixel_accuracy": self.evaluator.meanPixelAccuracy().cpu().numpy(),
            "mean_iou": self.evaluator.meanIntersectionOverUnion().cpu().numpy(),
            "frequency_weighted_iou": self.evaluator.frequency_weighted_iou(),
            "class_iou": self.evaluator.IntersectionOverUnion().cpu().numpy().tolist(),
            "class_pixel_accuracy": self.evaluator.classPixelAccuracy().cpu().numpy().tolist()
        } 
    
    def save_results_to_json(self) -> None:
        """将评估结果保存到JSON文件"""

        results = self.get_results()

        # 如果有类别名称，添加到结果中
        # if self.class_names:
        #     results["class_names"] = self.class_names
        #     results["class_iou"] = dict(zip(self.class_names, results["class_iou"]))
        #     results["class_pixel_accuracy"] = dict(zip(self.class_names, results["class_pixel_accuracy"]))
        
        if self.save_dir:
            save_path = os.path.join(self.save_dir, 'matrics.json')
            with open(save_path, 'w') as f:
                json.dump(results, f,default=to_serialize,indent=4)
    
    def visualize_result(self, image: torch.Tensor, label: torch.Tensor, prediction: torch.Tensor, 
                         class_names: Optional[List[str]] = None) -> None:
        """
        可视化语义分割结果
        Args:
            image: 输入图像 (C, H, W)
            label: 真实标签 (H, W)
            prediction: 预测结果 (C, H, W)
            class_names: 类别名称列表
            save_path: 保存路径, 默认为None(不保存)
        """
        # 确保输入是CPU张量
        image = image.cpu()
        label = label.cpu()
        prediction = prediction.cpu()
        
        # 获取预测类别
        predicted_class = torch.argmax(prediction, dim=0)
        
        # 创建可视化
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title("Original Image")
        plt.axis("off")
        
        # 真实标签
        plt.subplot(1, 3, 2)
        label_img = label.numpy()
        if class_names:
            label_img = np.array([class_names[i] for i in label_img], dtype=object)
        plt.imshow(label_img)
        plt.title("Ground Truth")
        plt.axis("off")
        
        # 预测结果
        plt.subplot(1, 3, 3)
        pred_img = predicted_class.numpy()
        if class_names:
            pred_img = np.array([class_names[i] for i in pred_img], dtype=object)
        plt.imshow(pred_img)
        plt.title("Prediction")
        plt.axis("off")
        
        # 保存或显示
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'visual_pre.png'))
        plt.close()

    def eval(self) -> None:
        """
        可视化并保存语义分割结果
        """        

        image,mask = next(iter(self.test_loader))
        image, mask = image.to(self.device), mask.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            predicted = output.squeeze(0)  # 移除批次维度
 
        self.save_results_to_json()
        self.visualize_result(
            image=image.squeeze(0), # 移除批次维度
            label=mask.squeeze(0), # 灰度图,删除通道维度
            prediction=predicted.cpu(),
        )

        