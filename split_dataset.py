# -*- coding: UTF-8 -*-
"""
*@ description: 分割数据集
*@ name:	split_dataset.py
*@ author: dengbozfan
*@ time:	2025/04/13 16:48
"""

from ProjectModel import bulid_dataset

if __name__ == '__main__':
    
    point_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PointSet"
    polygon_dataset_dir = ".\\NIH-NLM-ThinBloodSmearsPf\\PolygonSet"
    
    point_save_dir = ".\\resources\\PointSet"
    polygon_save_dir = ".\\resources\\PolygonSet"

    bulid_dataset(point_dataset_dir, save_dir=point_save_dir)

    bulid_dataset(polygon_dataset_dir, save_dir=polygon_save_dir)

