# -*- coding: UTF-8 -*-
"""
*@ description: 搭建数据集 | 分割数据集
*@ name:	bulid_dataset.py
*@ author: dengbozfan
*@ time:	2025/04/13 15:35
"""

from typing import Optional
from .. import os, ic, cv2
from tqdm import tqdm
from time import sleep

def bulid_dataset(dataset_dir : str, 
                  train_radio : int = 0.7,
                  save_dir: Optional[str] = None) -> None:
    """
    description: 分割数据集, 分割为train, test【数据集数量较少】
        制作txt文件只包含: 病人数据文件夹名称 病人图像名称【标签和图像名称相同】, 后续的读取在dataset中进行
        Args:
            dataset_dir : (str) 数据集目录,存储所有病人数据【病人数据依旧是文件夹】
            train_radio : (int) 训练数据集分割比例
            save_dir   : (Optional, str) txt文件存储目录
                若不输入,则存储置dataset_dir的同级目录中
        return:
            None, save txt文件
    """

    if not save_dir:
        save_dir = os.path.dirname(dataset_dir)
        print("数据集分割结果存储目录为: %s" % save_dir)

    with tqdm(total=100,desc="检测存储位置") as pbar:

        sleep(1)
        pbar.update(10)
        sleep(0.05)

        files_list = os.listdir(dataset_dir)
        dataSet_list = []

        for simple in tqdm(files_list,desc="搜集目录下的所有image name",leave=False):
            simple_path = os.path.join(dataset_dir,simple)
            simple_files = os.listdir(simple_path)
            img_path = os.path.join(simple_path,simple_files[1])
            imgs = os.listdir(img_path)
            for img in imgs:
                img_name = os.path.splitext(img)[0]
                image_file_path = os.path.join(dataset_dir,simple,"Img",f"{img_name}.jpg")
                image = cv2.imread(image_file_path)
                h,w = image.shape[:2]
                dataSet_list.append("%s %s %d %d" % (simple, img_name,h,w))
            
            sleep(0.01)

        pbar.update(60)
        sleep(0.05)

        pbar.set_description("开始分配数据集")
        sleep(0.05)

        train_dataset_index = int(len(dataSet_list)*train_radio)

        train_dataset = dataSet_list[:train_dataset_index]
        test_dataset = dataSet_list[train_dataset_index:]

        train_dateset_save_path = os.path.join(save_dir,"trainSet.txt")
        test_dataset_save_path = os.path.join(save_dir,"testSet.txt")
        pbar.update(80)
        sleep(0.05)

        pbar.set_description("开始将image name 写入txt")
        sleep(0.05)

        with open(train_dateset_save_path,'w') as f:
            for trainData in train_dataset:
                f.writelines(trainData+'\n')

        with open(test_dataset_save_path,'w') as f:
            for testData in test_dataset:
                f.writelines(testData+'\n')
                
        pbar.set_description("写入完成")
        pbar.update(100)
        sleep(0.05)

    print("dataset %s 分割完成😊" % dataset_dir)

