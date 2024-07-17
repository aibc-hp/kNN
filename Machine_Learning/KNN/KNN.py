# -*- coding: utf-8 -*-
# @Time    : 2023/11/24 10:16
# @Author  : aibc-hp
# @File    : KNN.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import numpy as np
import pandas as pd


# 读取数据集，并划分特征数据和标签数据
def read_dataset():
    df = pd.read_csv(r'D:\MachineLearning\movie_type.csv')  # 读取数据集

    data = df.iloc[:, 1:]  # 获取数据集的第 2、3、4 列数据
    data = data.to_numpy()  # 将 pandas.core.frame.DataFrame 转为 numpy.ndarray

    train_data = data[:, :2]  # data 的第 1、2 列为特征数据
    labels = data[:, -1]  # data 的第 3 列为标签数据

    return train_data, labels


# 计算距离
def calculate_distance(predict_data, train_data):
    dist = np.sqrt(np.sum((predict_data - train_data) ** 2, axis=1))  # 计算新样本数据与训练集中每一个样本数据间的距离

    return dist


# 预测结果
def select_best_result(dist, labels, k):
    labels_lst = [labels[index] for index in dist.argsort()[:k]]  # 获取前 k 个最相似数据对应的标签

    # 选取前 k 个标签中出现频率最高的作为最终结果
    num_labels = {}
    num = labels_lst.count(labels_lst[0])
    num_labels[labels_lst[0]] = num
    if len(labels_lst) > 1:
        for i in range(1, len(labels_lst)):
            for j in range(i):
                if labels_lst[i] == labels_lst[j]:
                    break
            else:  # 第二个循环没有执行 break 时，会执行 else
                num = labels_lst.count(labels_lst[i])
                num_labels[labels_lst[i]] = num

    result = max(num_labels, key=num_labels.get)  # 获取字典中每个键对应的值，并将最大值对应的键返回

    return result


if __name__ == '__main__':
    predict_data = np.array([101, 20])  # 预测数据

    train_data, labels = read_dataset()  # 获取特征数据和标签数据
    train_data = train_data.astype(float)  # 将整数数组转换为浮点数组，方便后续计算

    predict_data = np.full((4, 2), predict_data)  # 将预测数据填充为跟 train_data 有相同的维度
    predict_data = predict_data.astype(float)  # 将整数数组转换为浮点数组，方便后续计算

    dist = calculate_distance(predict_data, train_data)  # 计算距离

    result = select_best_result(dist, labels, k=1)  # 选取最好的结果

    print(result)
