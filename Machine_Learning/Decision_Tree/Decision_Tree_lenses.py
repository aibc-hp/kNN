# -*- coding: utf-8 -*-
# @Time    : 2023/11/29 19:58
# @Author  : aibc-hp
# @File    : Decision_Tree_lenses.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    """
    数据集中的数据都是 str 类型，无法直接作为 fit() 函数的参数，因此需要对数据进行转换
    通过 LabelEncoder 或 OneHotEncoder 编码可以将字符串转换为整数
    先将数据集数据转换成 pandas 数据，方便后续操作，原始数据 -> 字典 -> DataFrame
    """
    file_path = r'D:\MachineLearning\lenses.txt'

    # 读取数据
    with open(file_path, 'r') as file:
        lenses = [line.strip().split('\t') for line in file.readlines()]

    # 提取样本标签
    lenses_targets = []
    for sample in lenses:
        lenses_targets.append(sample[-1])

    feature_labels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 定义特征标签

    # 将特征存放到字典
    lenses_dict = {}  # 以键值对方式存储特征标签与特征属性值
    for label in feature_labels:
        lenses_list = []  # 存储某一特征标签对应的特征属性值
        for sample in lenses:
            lenses_list.append(sample[feature_labels.index(label)])
        lenses_dict[label] = lenses_list

    lenses_pd = pd.DataFrame(lenses_dict)  # 将字典转换成 DataFrame

    # 将字符串转换成整数
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    clf = DecisionTreeClassifier()  # 实例化 DecisionTreeClassifier 类对象
    clf.fit(lenses_pd.values.tolist(), lenses_targets)  # 拟合

    test_data = [[1, 1, 1, 0],
                 [2, 1, 1, 1]]
    predict_result = clf.predict(test_data)  # 预测结果

    print(predict_result)

