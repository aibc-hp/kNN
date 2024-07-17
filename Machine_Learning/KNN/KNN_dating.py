# -*- coding: utf-8 -*-
# @Time    : 2023/11/24 15:51
# @Author  : aibc-hp
# @File    : KNN_dating.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import numpy as np
import pandas as pd


# 读取数据集，将数据集划分成训练集和测试集，并划分特征数据和标签数据，同时将标签进行相应转换以方便后续处理
def read_dataset():
    df = pd.read_table(r'D:\MachineLearning\dating_set.txt', header=None)  # 读取数据集，共 1000 个样本

    data = df.iloc[:, :]  # 获取数据集的第 1、2、3、4 列数据

    train_for_data = data.sample(frac=0.9)  # 从原始数据 data 中随机选择 90% 的数据作为训练集
    test_for_data = data.drop(train_for_data.index)  # 从原始数据 data 中提取剩下的 10% 数据作为测试集
    train_for_data = train_for_data.to_numpy()  # 将 pandas.core.frame.DataFrame 转为 numpy.ndarray
    test_for_data = test_for_data.to_numpy()  # 将 pandas.core.frame.DataFrame 转为 numpy.ndarray

    train_data = train_for_data[:, :3]  # train_for_data 的第 1、2、3 列为训练集的特征数据
    train_labels = train_for_data[:, -1]  # train_for_data 的第 4 列为训练集的标签数据
    test_data = test_for_data[:, :3]  # test_for_data 的第 1、2、3 列为训练集的特征数据
    test_labels = test_for_data[:, -1]  # test_for_data 的第 4 列为训练集的标签数据

    label_mapping = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}  # 建立能将字符串标签映射成数字标签的字典

    train_labels = np.array([label_mapping[label] for label in train_labels])  # 将字符串标签转换成数字标签
    test_labels = np.array([label_mapping[label] for label in test_labels])  # 将字符串标签转换成数字标签

    return train_data, test_data, train_labels, test_labels


# 归一化
def normalize(train_data, test_data):
    for i in range(train_data.shape[1]):
        arr = train_data[:, i]  # 一列特征数据

        max_value = arr.max()  # 最大值
        min_value = arr.min()  # 最小值
        arr = (arr - min_value) / (max_value - min_value)  # 归一化计算

        train_data[:, i] = arr

    for i in range(test_data.shape[1]):
        arr = test_data[:, i]  # 一列特征数据

        max_value = arr.max()  # 最大值
        min_value = arr.min()  # 最小值
        arr = (arr - min_value) / (max_value - min_value)  # 归一化计算

        test_data[:, i] = arr

    return train_data, test_data


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


# 计算错误率
def calculate_error_rate(test_result, test_labels):
    num_error = 0
    for i in range(len(test_result)):
        if test_result[i] != test_labels[i]:
            num_error += 1

    error_rate = num_error / len(test_result) * 100

    print(f'错误率为：{error_rate}%')


if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels = read_dataset()  # 获取用于训练与测试的特征数据和标签数据

    train_data, test_data = normalize(train_data, test_data)  # 将用于训练与测试的特征数据归一化
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    num_samples = train_data.shape[0]  # 训练集中的样本个数（行数）
    num_features = train_data.shape[1]  # 训练集中的特征个数（列数）

    test_result = []
    for i in range(len(test_data)):
        predict_data = np.full((num_samples, num_features), test_data[i])  # 将测试数据集中的一个样本填充为跟 train_data 有相同的维度
        predict_data = predict_data.astype(float)

        dist = calculate_distance(predict_data, train_data)  # 计算距离

        result = select_best_result(dist, train_labels, k=1)  # 选取最好的结果

        test_result.append(result)

    test_result = np.array(test_result)

    calculate_error_rate(test_result, test_labels)  # 计算测试集的错误率



