# -*- coding: utf-8 -*-
# @Time    : 2023/11/28 9:51
# @Author  : aibc-hp
# @File    : Decision_Tree.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import math
import operator
import copy
import pickle


# 划分数据子集，并返回某一特征列中的某一类别子集
def split_dataset(dataset: list, feature_num: int, key: any) -> list:
    """
    :param dataset: 整个数据集
    :param feature_num: 特征列的索引
    :param key: 特征列的某一类别
    :return: 依据特征列中某一类别划分的子数据集
    """
    sub_dataset = []
    for sample in dataset:
        if sample[feature_num] == key:
            sub_dataset.append(sample)

    return sub_dataset


# 划分数据子集，并返回某一特征列中的某一类别子集，该子集删除了对应特征
def split_del_dataset(dataset: list, feature_num: int, key: any) -> list:
    """
    :param dataset: 整个数据集
    :param feature_num: 特征列的索引
    :param key: 特征列的某一类别
    :return: 依据特征列中某一类别划分的子数据集（删除了对应特征）
    """
    sub_dataset = []
    for sample in dataset:
        if sample[feature_num] == key:
            sample.pop(feature_num)
            sub_dataset.append(sample)

    return sub_dataset


# 判定主要类别
def primary_category(label_targets: list) -> str:
    """
    :param label_targets: 标签列
    :return: 返回出现次数最多的类别
    """
    label_targets_dict = {}
    for label in label_targets:
        if label not in label_targets_dict.keys():
            label_targets_dict[label] = 0
        label_targets_dict[label] += 1

    sorted_label_targets = sorted(label_targets_dict.items(), key=operator.itemgetter(1), reverse=True)  # 将键值对以元组形式按值的大小进行从大到小排列

    return sorted_label_targets[0][0]


# 计算数据集或特征列的熵
def calculate_entropy(dataset: list, feature_num: int) -> float:
    """
    :param dataset: 整个数据集
    :param feature_num: 特征列或目标列的索引
    :return: 数据集或特征列的熵
    """
    num_samples = len(dataset)  # 数据集的样本个数

    # 将特征列或目标列的类别及出现次数存储到字典
    targets = {}  # 用于存储标签出现的次数
    for sample in dataset:
        if sample[feature_num] not in targets.keys():
            targets[sample[feature_num]] = 0  # 初始化标签次数
        targets[sample[feature_num]] += 1  # 标签次数 +1

    # 计算特征列或目标列的熵
    entropy = 0.0
    for key in targets:
        prob = targets[key] / num_samples  # 特征列或目标列中某一类别出现的概率
        entropy -= prob * math.log(prob, 2)  # 计算熵

    return entropy


# 计算特征列的条件熵
def calculate_conditional_entropy(dataset: list, feature_num: int) -> float:
    """
    :param dataset: 整个数据集
    :param feature_num: 特征列的索引
    :return: 特征列的条件熵
    """
    num_samples = len(dataset)  # 数据集的样本个数

    # 将特征列的类别及出现次数存储到字典
    feature_targets = {}  # 用于存储特征列标签出现的次数
    for sample in dataset:
        if sample[feature_num] not in feature_targets.keys():
            feature_targets[sample[feature_num]] = 0  # 初始化标签次数
        feature_targets[sample[feature_num]] += 1  # 标签次数 +1

    # 计算特征列的条件熵
    conditional_entropy = 0.0
    for key in feature_targets:
        sub_dataset = split_dataset(dataset, feature_num, key)  # 获取 feature_num 特征列中的 key 类别子集

        prob = feature_targets[key] / num_samples  # 特征列中某一类别出现的概率
        entropy = calculate_entropy(sub_dataset, -1)
        conditional_entropy += prob * entropy

    return conditional_entropy


# 计算信息增益
def information_gain(entropy: float, conditional_entropy: float) -> float:
    """
    :param entropy: 数据集的熵
    :param conditional_entropy: 特征列的条件熵
    :return: 特征的信息增益
    """
    ig = entropy - conditional_entropy

    return ig


# 计算所有特征的信息增益，并选择信息增益最大的特征作为最佳特征
def choose_best_feature(dataset: list) -> int:
    """
    :param dataset: 整个数据集
    :return: 信息增益最大的特征的索引
    """
    num_features = len(dataset[0]) - 1  # 特征个数
    ig_lst = []  # 用于存储所有特征的信息增益

    entropy = calculate_entropy(dataset, -1)  # 计算数据集的熵

    # 遍历计算所有特征的信息增益
    for idx in range(num_features):
        conditional_entropy = calculate_conditional_entropy(dataset, idx)  # 计算索引为 idx 的特征的条件熵
        ig = information_gain(entropy, conditional_entropy)  # 计算索引为 idx 的特征的信息增益
        ig_lst.append(ig)  # 将当前特征的信息增益添加到存储列表

    best_feature_idx = ig_lst.index(max(ig_lst))  # 获取信息增益最大的特征的索引

    return best_feature_idx


# 读取数据集
def create_dataset() -> (list, list):
    """
    :return: 返回数据集和特征名称
    """
    dataset = [[1, 2, 2, 1, 'no'],
               [1, 2, 2, 2, 'no'],
               [1, 1, 2, 2, 'yes'],
               [1, 1, 1, 1, 'yes'],
               [1, 2, 2, 1, 'no'],
               [2, 2, 2, 1, 'no'],
               [2, 2, 2, 2, 'no'],
               [2, 1, 1, 2, 'yes'],
               [2, 2, 1, 3, 'yes'],
               [2, 2, 1, 3, 'yes'],
               [3, 2, 1, 3, 'yes'],
               [3, 2, 1, 2, 'yes'],
               [3, 1, 2, 2, 'yes'],
               [3, 1, 2, 3, 'yes'],
               [3, 2, 2, 1, 'no']]

    feature_labels = ['年龄', '是否有工作', '是否有房子', '信贷情况']

    return dataset, feature_labels


# 构建决策树
def create_decision_tree(dataset: list, feature_labels: list, best_feature_labels: list) -> dict:
    """
    :param dataset: 整个数据集
    :param feature_labels: 存放了所有特征的名称
    :param best_feature_labels: 用于存放最佳特征的名称
    :return: 以字典形式返回生成的决策树及以列表形式返回最佳特征
    """
    label_targets = [sample[-1] for sample in dataset]  # 数据集的目标列（是否提供贷款，yes 或 no）

    # 判断数据集的目标列中类别是否完全相同，即只有一个类别，并返回该类别
    if label_targets.count(label_targets[0]) == len(label_targets):
        return label_targets[0]

    # 判断是否遍历完所有特征，并返回出现次数最多的类别
    if len(dataset[0]) == 1:
        return primary_category(label_targets)

    best_feature_idx = choose_best_feature(dataset)  # 获取当前数据集最佳特征的索引
    best_feature_name = feature_labels[best_feature_idx]  # 获取当前数据集最佳特征的名称
    best_feature_labels.append(best_feature_name)  # 将当前数据集最佳特征的名称添加到列表

    decision_tree = {best_feature_name: {}}  # 根据最佳特征的名称生成树
    del feature_labels[best_feature_idx]  # 删除已经使用过的特征名称

    best_feature_values = [sample[best_feature_idx] for sample in dataset]  # 获取最佳特征的所有属性值
    unique_feature_values = set(best_feature_values)  # 去掉重复的属性值，剩下的就是类别

    # 遍历特征，构造决策树
    for value in unique_feature_values:
        new_dataset = copy.deepcopy(dataset)  # 这里需要使用深拷贝创建一个新的完全相同的数据集，否则后面的改动会影响原始数据集
        sub_dataset = split_del_dataset(new_dataset, best_feature_idx, value)
        decision_tree[best_feature_name][value] = create_decision_tree(sub_dataset, feature_labels, best_feature_labels)

    return decision_tree


# 利用决策树进行分类预测
def classify_predict(decision_tree: dict, best_feature_labels: list, test_data: list) -> str:
    """
    :param decision_tree: 当前决策树
    :param best_feature_labels: 存放了最佳特征的名称
    :param test_data: 预测数据
    :return: 返回预测标签
    """
    root_node = next(iter(decision_tree))  # decision_tree 中的第一个键就是根节点，也就是第一个最佳特征
    root_node_value = decision_tree[root_node]  # 获取当前根节点的值，是一个字典或字符串

    predict_result = 'none'
    if root_node in best_feature_labels:
        idx = best_feature_labels.index(root_node)
        for key in root_node_value.keys():
            if test_data[idx] == key:
                if type(root_node_value[key]).__name__ == 'str':
                    predict_result = root_node_value[key]
                else:
                    predict_result = classify_predict(root_node_value[key], best_feature_labels, test_data)

    return predict_result


# 存储决策树
def store_decision_tree(decision_tree, filename):
    with open(filename, 'wb') as file:
        pickle.dump(decision_tree, file)


# 调用决策树
def obtain_decision_tree(filename):
    with open(filename, 'rb') as file:
        decision_tree = pickle.load(file)

    return decision_tree


if __name__ == '__main__':
    dataset, feature_labels = create_dataset()  # 获取数据集和特征名称
    best_feature_labels = []
    decision_tree = create_decision_tree(dataset, feature_labels, best_feature_labels)  # 生成决策树
    test_data = [1, 1]  # 预测数据，维数取决于决策树的内部节点数
    predict_result = classify_predict(decision_tree, best_feature_labels, test_data)

    print(decision_tree)
    print(predict_result)


