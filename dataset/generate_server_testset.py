"""
Author: Xinpeng Ling
Target: Coding for Put the test set of local together to become the test set of server.
Email:  xpling@stu.ecnu.edu.cn
Home page: https://space.bilibili.com/3461572290677609
"""
import numpy as np


def generate_server_testset(test_data, test_path):
    """ 作用是把test_data并到一起，给存到test_path """
    # 创建一个新的字典用于存储合并后的数据
    merged_dict = {'x': [], 'y': []}
    # 循环遍历字典列表
    for d in test_data:
        for key, value in d.items():
            merged_dict[key].extend(value.tolist())
    # 将列表中的数据转换为numpy数组
    merged_dict['x'] = np.array(merged_dict['x'])
    merged_dict['y'] = np.array(merged_dict['y'])
    with open(test_path + "server_testset.npz", 'wb') as f:
        np.savez_compressed(f, data=merged_dict)
