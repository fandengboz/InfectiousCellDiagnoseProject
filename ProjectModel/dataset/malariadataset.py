# -*- coding: UTF-8 -*-
"""
*@ description: 疟疾数据集加载Dataset
*@ name:	Dataset.py
*@ author: dengbozfan
*@ time:	2025/04/12 22:01
"""

from .. import torch,Dataset,transforms
from .. import os,np,cv2,singledispatch
from .. import closest_power_of_two

from typing import Optional,Callable,Union,Tuple

@singledispatch
def check(obj):
    raise NotImplementedError(f"Cannot process object of type {type(obj)}")

@check.register
def _(obj:None):

    obj = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
    ])
    return obj

@check.register
def _(obj:transforms.Compose):
    return obj


class MalariaDataset(Dataset):

    def __init__(
            self, 
            image_dir : str, 
            dataset_path : str, 
            transform : Optional[Callable] = None,
            resize_zoom : float = 1.0
        ):
        """
        初始化 MalariaDataset 类。

        Args:
            image_dir (str): 图像文件的目录路径。
            dataset_path (str): 存储分割后的数据 txt文件路径, 包含【病人名称 图片名称】
            transform (callable, optional): 可选的图像变换函数。默认为 None。
            resize_zoom (float): 缩放因子, 对于较大的图片进行缩放
        """
        self.image_dir = image_dir
        self.dataset_path = dataset_path
        self.transform = check(transform)
        self.resize_zoom = resize_zoom 

        self.image_files = []
        self.annotation_files = []
        self.image_sizes = []  # 存储每张图片的原始尺寸

        self.get_file_name_list()
        self.uniform_size = self.calculate_uniform_size()

    def get_file_name_list(self):
        """获取名称列表"""

        with open(self.dataset_path,'r') as files:
            for line in files.readlines():
                dir_name, image_name,h,w = line.strip().split()

                image_file_path = os.path.join(self.image_dir,dir_name,"Img",f"{image_name}.jpg")
                annotation_file_path = os.path.join(self.image_dir,dir_name,"GT",f"{image_name}.txt")

                #获取图片尺寸
                self.image_sizes.append((int(w),int(h)))
                self.image_files.append(image_file_path)
                self.annotation_files.append(annotation_file_path)

    def calculate_uniform_size(self) -> Union[int, Tuple[int, int]] :
        """计算统一的尺寸"""

        if not self.image_sizes:
            return (256, 256)  # 默认尺寸
        
        # 计算每张图片最接近的2的幂次尺寸
        power_sizes = [closest_power_of_two(size) for size in self.image_sizes]
        
        # 找到最大宽度和高度
        max_width = max(size[0] for size in power_sizes) * self.resize_zoom
        max_height = max(size[1] for size in power_sizes) * self.resize_zoom
        
        # 确保最大尺寸也是2的幂次
        return closest_power_of_two((max_width, max_height))

    def resize_image(self,image:np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """调整图片大小"""

        assert type(image) == np.ndarray
        assert type(mask) == np.ndarray

        new_width, new_height = self.uniform_size
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return image, mask
    

    def __len__(self) -> int:
        
        pass

    def __getitem__(
            self, 
            index: int):
        
        print("%d" % index)

        pass

    def __repr__(self):
        """打印讯息"""
        image_dir_info = F"image_dir : {self.image_dir}"
        dataset_path_info = F"dataset_path: {self.dataset_path}"
        len_images_info = F"len images : {len(self.image_files)}"
        len_annotations_info = F"len annotations : {len(self.annotation_files)}"

        information = F"{image_dir_info}\n{dataset_path_info}\n{len_images_info}\n{len_annotations_info}"
        
        return information
    

    
    