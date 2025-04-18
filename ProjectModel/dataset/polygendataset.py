# -*- coding: UTF-8 -*-
"""
*@ description: 多边形集 Dataset | 分割细胞 |
*@ name:	polygendataset.py
*@ author: dengbozfan
*@ time:	2025/04/13 21:10
"""

from .malariadataset import MalariaDataset
from .. import os, torch, cv2,np,normalize

class PolygonDataset(MalariaDataset):

    def __init__(self, image_dir, dataset_path, transform = None, resize_zoom = 1):
        super().__init__(image_dir, dataset_path, transform, resize_zoom)
            
    def __len__(self):

        return len(self.image_files)
    
    def __getitem__(self, index):
        """
        description: 返回指定索引处的数据
            Args: index : (int) 索引
            return: image, mask
        """

        image_path = self.image_files[index]
        annotation_path = self.annotation_files[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 
        with open(annotation_path,'r') as f:
            lines = f.readlines()
            mask = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
            for line in lines[1:]:
                parts = line.strip().split(',')
                assert parts[3] == "Polygon" , '数据集类别不是 Polygon'
                # 提取点集
                points = np.array(parts[5:],dtype=np.float32).reshape(-1,2)
                points = points.astype(np.int32)  # 转换为 int32 类型
                cv2.fillPoly(mask,[points],1)

        # 将图像和掩码调整为同一的尺寸
        image, mask = self.resize_image(image,mask)
        
        # 归一化
        image = normalize(image)

        # 应用变换
        image = self.transform(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    

