# -*- coding: UTF-8 -*-
"""
*@ description:
*@ name:	trainModel.py
*@ author: dengbozfan
*@ time:	2025/04/17 20:30
"""

from ProjectModel import bulid_dataset,DiceLoss
from ProjectModel import PointDataset,PolygonDataset
from ProjectModel import UNet_up_Resnet18,Unet
from ProjectModel import Train,PlotResult,MalariaEvaluator

import torch.nn as nn
import torch.optim as optim

from icecream import ic

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from functools import singledispatch

if __name__ == '__main__':
    
    polygon_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PolygonSet"

    trainSet_dir = ".\\resources\\PolygonSet\\trainSet.txt"
    valSet_dir = ".\\resources\\PolygonSet\\valSet.txt"
    testSet_dir = ".\\resources\\PolygonSet\\testSet.txt"

    # trainSet_dir = ".\\resources\\examineSet\\trainSet.txt"
    # valSet_dir = ".\\resources\\examineSet\\valSet.txt"
    # testSet_dir = ".\\resources\\examineSet\\testSet.txt"


    train_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=trainSet_dir,resize_zoom=0.6)
    val_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=valSet_dir,resize_zoom=0.6)
    test_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=testSet_dir,resize_zoom=0.6)

    BATCH_SIZE = 1
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = UNet_up_Resnet18(n_classes=2)
    model = Unet(3,2)
    
    run_dir = ".\\resources\\runs"
    tra = Train(model,2,train_loader,test_loader,val_loader,run_dir=run_dir)
    
    loss_func = DiceLoss(num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    tra.train(num_epochs=100,optimizer=optimizer,scheduler=scheduler,loss_func=loss_func,save_checkpoint=True,
              checkpoint_interval=5,save_best_model=True)
    
    