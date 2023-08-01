# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:31:20 2021

@author: yangzhen
"""

def _init_():#初始化
    global _global_dict
    _global_dict = {}
 
 
def set_value(key,value):
    """ 定义一个全局变量 """
    _global_dict[key] = value
 
 
def get_value(key,defValue=None):
# 　　""" 获得一个全局变量,不存在则返回默认值 """
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
