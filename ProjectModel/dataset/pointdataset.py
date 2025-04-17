# -*- coding: UTF-8 -*-
"""
*@ description: 点集 Dataset
*@ name:	pointdataset.py
*@ author: dengbozfan
*@ time:	2025/04/13 21:10
"""

from .malariadataset import MalariaDataset
from .. import os, torch, cv2


class PointDataset(MalariaDataset):

    def __init__(self, image_dir, annotation_dir, transform = None):
        """
        ['1-1', 'Parasitized', 'No_Comment', 'Point', '1', '3833.3', '306.6']
        """
        
        super().__init__(image_dir, annotation_dir, transform)

    def __len__(self):

        return len(self.image_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] :
        """
        description: 返回指定索引处的数据
            Args: index : (int) 索引
            return: image, bboxes[torch.float32]
        """
        image_path = self.image_files[index]
        annotation_path = self.annotation_files[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(annotation_path,'r') as f:
            lines = f.readlines()
            bboxes = []
            
            for line in lines[1:]:
                parts = line.strip().split(',')
                assert parts[3] == "Point" , '数据集类别不是 Point '
                
                x,y = int(parts[5]),int(parts[6])
                # 通过中心点设置为一个检测框
                bboxes.append([x,y,x+10,y+10])

        if self.transform:
           image = self.transform(image)

        return image, torch.tensor(bboxes,dtype=torch.float32)
    


        
       




        