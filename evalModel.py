# -*- coding: UTF-8 -*-
"""
*@ description: 检验模型效果
*@ name:	evalModel.py
*@ author: dengbozfan
*@ time:	2025/04/18 19:21
"""

from ProjectModel import bulid_dataset,DiceLoss
from ProjectModel import PointDataset,PolygonDataset
from ProjectModel import UNet_up_Resnet18,Unet
from ProjectModel import Train,PlotResult,MalariaEvaluator
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    polygon_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PolygonSet"
    testSet_dir = ".\\resources\\PolygonSet\\testSet.txt"
    test_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=testSet_dir,resize_zoom=0.6)

    BATCH_SIZE = 1
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = UNet_up_Resnet18(n_classes=2)
    model = Unet(3,2)
    model.to(DEVICE)
    image,mask = test_dataset[0]
    
    best_model_path = ".\\resources\\runs\\UNetSeg\\model\\best.pt"
    save_dir=".\\resources\\runs\\UNetSeg\\eval"
    evaluator = MalariaEvaluator(num_classes=2,model=model,test_loader=test_loader,
                                 device=DEVICE,save_dir=save_dir,best_model_path=best_model_path)
    ma = evaluator.eval()