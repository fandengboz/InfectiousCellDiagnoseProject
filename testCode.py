# -*- coding: UTF-8 -*-
"""
*@ description: 测试代码的文件
*@ name:	testCode.py
*@ author: dengbozfan
*@ time:	2025/04/13 15:46
"""

from ProjectModel import bulid_dataset
from ProjectModel import PointDataset,PolygonDataset
from ProjectModel import UNet_up_Resnet18
from ProjectModel import Train,PlotResult,MalariaEvaluator

import torch.nn as nn
import torch.optim as optim

from icecream import ic

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from functools import singledispatch


# from ProjectModel import closest_power_of_two

if __name__ == '__main__':
    
    # point_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PointSet"
    # annotation_dir = ".\\resources\\PointSet\\trainSet.txt"

    # train_dataset = PointDataset(image_dir=point_dataset_dir,annotation_dir=annotation_dir)

    # print(train_dataset[0])

    polygon_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PolygonSet"
    # trainSet_dir = ".\\resources\\PolygonSet\\trainSet.txt"
    # testSet_dir = ".\\resources\\PolygonSet\\testSet.txt"

    # 测验数据
    trainSet_dir = ".\\resources\\examineSet\\trainSet.txt"
    testSet_dir = ".\\resources\\examineSet\\testSet.txt"

    train_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=trainSet_dir,resize_zoom=0.7)
    test_dataset = PolygonDataset(image_dir=polygon_dataset_dir,dataset_path=testSet_dir,resize_zoom=0.7)

    BATCH_SIZE = 1
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet_up_Resnet18(n_classes=2)
    # print(type(model))
    model.to(DEVICE)

    run_dir = ".\\resources\\runs"
    tra = Train(model,train_loader,test_loader,run_dir=run_dir)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tra.train(num_epochs=10,optimizer=optimizer,loss_func=loss_func,save_checkpoint=True,
              checkpoint_interval=1,save_best_model=True)

    # save_dir = ".\\resources\\runs\\run6\\config"
    # train_log_path = ".\\resources\\runs\\run6\\model\\trainLog.csv"
    # plot_r = PlotResult(save_dir=save_dir)
    # plot_r.compute_and_plot_confusion_matrix(1,model=model,test_loader=test_loader,device=DEVICE)
    # plot_r.plot_acc(train_log_path,save_dir)
    # plot_r.plot_loss(train_log_path,save_dir)
    # plot_r.plot_current_lr(train_log_path,save_dir)
    # image,mask = train_dataset[0]
    # evaluator = MalariaEvaluator(num_classes=2,model=model,test_loader=test_loader,
    #                              device=DEVICE,save_dir=".\\resources\\runs\\run6\\eval")
    # ma = evaluator.eval()
    # evaluator.eval(model,image,mask,class_names=["0","1"],device=DEVICE,
    #                )
    
    # # 输入张量 (batch_size, channels, height, width)
    # input_tensor, mask_tensor = next(iter(train_loader))

    # input_tensor = torch.randn(1, 3, 256, 256)
    # # # print(input_tensor.shape)
    # # # output = backbone(input_tensor)
    # output = model(input_tensor.to(DEVICE))
    # print(output.shape)  # 输出张量 (batch_size, num_classes, height, width)

    # # 示例使用
    # print(closest_power_of_two(250))  # 输出: 256
    # print(closest_power_of_two(512))  # 输出: 512
    # print(closest_power_of_two((300, 400)))  # 输出: (256, 512)
    
    # transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.ConvertImageDtype(torch.float),
    # ])

    # @singledispatch
    # def check(obj):
    #     raise NotImplementedError(f"Cannot process object of type {type(obj)}")
    
    # @check.register
    # def _(obj:None):
    #     obj = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.ConvertImageDtype(torch.float),
    #     ])
    #     return obj
            
    # # print(type(transform))

    # print(check(None))


    # polygon_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PolygonSet"
    # output_dir = ".\\resources\\PolygonSet"
    # bulid_dataset(dataset_dir=polygon_dataset_dir,save_dir=output_dir)



