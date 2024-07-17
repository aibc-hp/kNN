# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 9:17
# @Author  : aibc-hp
# @File    : KNN_handwritten-digit-recognition.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# 将 (32, 32) 的矩阵转换成 (1, 1024) 的向量
def mat_to_vector(file: str) -> np.ndarray:
    df = pd.read_table(file, header=None)
    df = df.to_numpy()
    vec = np.zeros((1, df.shape[0] * df.shape[0]))  # (1, 1024)

    with open(file, 'r') as f:
        rows = f.readlines()  # 读取文件中的所有行，并以列表形式返回
        for i in range(len(rows)):
            row = rows[i].strip()  # 读取列表中的一个字符串元素
            columns = [int(row[i:i+1]) for i in range(len(row))]  # 将字符串分割成单个数字，并以列表形式返回
            for j in range(len(columns)):
                vec[0, 32 * i + j] = int(columns[j])  # 将每一个数字赋值给向量 vec 对应的位置

    return vec


# 读取训练集
def read_train_dataset(path: str) -> (np.ndarray, np.ndarray):
    train_labels = []  # 用于存储手写数字图像对应的数字标签

    train_files = os.listdir(path)  # 读取所有二进制图像文件，并以列表形式返回
    df = pd.read_table(os.path.join(path, train_files[0]), header=None)
    df = df.to_numpy()
    m = len(train_files)  # 1934
    train_mat = np.zeros((m, df.shape[0] * df.shape[0]))  # (1934, 1024)

    for i in range(m):
        train_file_name = train_files[i]
        digit = int(train_file_name.split('_')[0])
        train_labels.append(digit)  # 将每一个图像文件对应的数字标签存储到列表
        train_mat[i, :] = mat_to_vector(os.path.join(path, train_files[i]))  # 将每一个 (1, 1024) 的二进制图像数据赋值到矩阵

    train_labels = np.array(train_labels)

    return train_mat, train_labels


# 读取测试集
def read_test_dataset(path: str) -> (np.ndarray, np.ndarray):
    test_labels = []  # 用于存储手写数字图像对应的数字标签

    test_files = os.listdir(path)  # 读取所有二进制图像文件，并以列表形式返回
    df = pd.read_table(os.path.join(path, test_files[0]), header=None)
    df = df.to_numpy()
    m = len(test_files)  # 946
    test_mat = np.zeros((m, df.shape[0] * df.shape[0]))  # (946, 1024)

    for i in range(m):
        test_file_name = test_files[i]
        digit = int(test_file_name.split('_')[0])
        test_labels.append(digit)  # 将每一个图像文件对应的数字标签存储到列表
        test_mat[i, :] = mat_to_vector(os.path.join(path, test_files[i]))  # 将每一个 (1, 1024) 的二进制图像数据赋值到矩阵

    test_labels = np.array(test_labels)

    return test_mat, test_labels


# 构建 KNN 模型
def knn_model(train_data: np.ndarray, train_labels: list) -> object:
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, train_labels)

    return neigh


if __name__ == '__main__':
    train_path = r'D:\MachineLearning\trainingDigits'
    test_path = r'D:\MachineLearning\testDigits'

    train_data, train_labels = read_train_dataset(train_path)  # 读取训练数据，并返回训练集和对应标签

    neigh = knn_model(train_data, train_labels)  # 构建 KNN 模型，并返回 KNN 对象

    test_data, test_labels = read_test_dataset(test_path)  # 读取测试数据，并返回测试集和对应标签
    print(neigh.score(test_data, test_labels))

    result = neigh.predict(test_data)  # 预测结果，并以 np.ndarray 形式返回

    result_lst = (result - test_labels).tolist()  # 将数组转成列表

    error_rate = (len(result_lst) - result_lst.count(0)) / len(result_lst) * 100  # 计算错误率

    print(f'错误率为：{error_rate}%')


