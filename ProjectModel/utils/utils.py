# -*- coding: UTF-8 -*-
"""
*@ description: 工具函数
*@ name:	utils.py
*@ author: dengbozfan
*@ time:	2025/04/16 20:25
"""
from .. import np
from functools import singledispatch
from typing import Any, Dict, List, Union

@singledispatch
def to_serialize(obj: Any) -> Any:
    """通用序列化函数"""
    raise TypeError(f"Type {type(obj)} not serializable")

@to_serialize.register(np.ndarray)
def _(arr: np.ndarray) -> List:
    """序列化 NumPy 数组"""
    return arr.tolist()

@to_serialize.register(np.float32)
@to_serialize.register(np.float64)
def _(scalar: Union[np.float32, np.float64]) -> float:
    """序列化 NumPy 浮点数"""
    return float(scalar)

@to_serialize.register(np.int32)
@to_serialize.register(np.int64)
def _(scalar: Union[np.int32, np.int64]) -> int:
    """序列化 NumPy 整数"""
    return int(scalar)

@to_serialize.register(list)
def _(lst: list) -> list:
    """序列化列表"""
    return [to_serialize(item) for item in lst]

@to_serialize.register(dict)
def _(dct: dict) -> dict:
    """序列化字典"""
    return {key: to_serialize(value) for key, value in dct.items()}

@to_serialize.register(str)
@to_serialize.register(int)
@to_serialize.register(float)
def _(obj: Union[str, int, float]) -> Union[str, int, float]:
    """直接返回 Python 基本类型"""
    return obj