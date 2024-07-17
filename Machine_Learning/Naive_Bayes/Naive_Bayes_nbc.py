# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:35
# @Author  : aibc-hp
# @File    : Naive_Bayes_nbc.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import os
import random
import jieba
import numpy as np
from sklearn.naive_bayes import MultinomialNB


# 数据集处理
def text_process(dir_path: str, train_size=0.8) -> (list, list, list, list, list):
    """
    :param dir_path: 数据集目录
    :param train_size: 从数据集中划分训练集的比例
    :return: 词汇表，训练数据，测试数据，训练标签，测试标签
    """
    dir_list = os.listdir(dir_path)  # ['C000008', ...]

    data_list = []
    labels_list = []

    # 遍历每个存放了 txt 文件的子目录
    for dir in dir_list:
        new_dir_path = os.path.join(dir_path, dir)
        files = os.listdir(new_dir_path)  # ['10.txt', ...]

        # 遍历每个存储了新闻文本的 txt 文件
        for file in files:
            file_path = os.path.join(new_dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的生成器
            word_list = list(word_cut)

            data_list.append(word_list)
            labels_list.append(dir)

    # 划分训练集与测试集
    data_labels_list = list(zip(data_list, labels_list))  # 将数据与标签对应压缩
    random.shuffle(data_labels_list)  # 将 data_labels_list 乱序
    index = int(len(data_labels_list) * train_size) + 1  # 训练集与测试集划分的索引值
    train_list = data_labels_list[:index]  # 训练集，包括数据与标签
    test_list = data_labels_list[index:]  # 测试集，包括数据与标签
    train_data_list, train_labels_list = zip(*train_list)  # 解压训练集，得到训练数据和标签
    train_data_list, train_labels_list = list(train_data_list), list(train_labels_list)  # 转成列表
    test_data_list, test_labels_list = zip(*test_list)  # 解压测试集，得到测试数据和标签
    test_data_list, test_labels_list = list(test_data_list), list(test_labels_list)  # 转成列表

    # 统计数据集词频
    all_words_dict = {}
    for words in train_data_list:
        for word in words:
            if word not in all_words_dict.keys():
                all_words_dict[word] = 0
            all_words_dict[word] += 1

    # 根据字典中的值进行键值对的排序，排列顺序为降序
    all_words_zip = sorted(all_words_dict.items(), key=lambda x: x[1], reverse=True)  # 排序
    all_words_tuple, all_words_frequency_tuple = zip(*all_words_zip)  # 解压缩，得到元组形式的词汇表和频次表
    all_words_list = list(all_words_tuple)  # 转成列表

    return all_words_list, train_data_list, test_data_list, train_labels_list, test_labels_list


# 一些特定的词语如“的”、“在”、“当然”等对新闻分类无实际意义，将这些词整理好并存储在了 stopwords_cn.txt 文件中
# 读取 stopwords_cn.txt 文件，并进行去重处理
def stop_words_set(file_path: str) -> set:
    """
    :param file_path: stopwords_cn.txt 的路径
    :return: 返回一个经过去重处理的词汇集合
    """
    words_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)

    return words_set


# 词频最高的往往是一些对于分类无意义的符号，有必要删除它们
# 文本特征选取，删除词频最高的 n 个词，并选取合适的词作为特征词
def delete_words(all_words_list: list, stopwords_set: set, n=10) -> list:
    """
    :param all_words_list: 训练集的词汇表
    :param stopwords_set: 无意义的词汇集合
    :param n: 要删除的前多少个高频词汇数
    :return: 特征词汇表
    """
    feature_words = []
    for i in range(n, len(all_words_list)):
        if all_words_list[i].isdigit() or all_words_list[i] in stopwords_set or len(all_words_list[i]) <= 1 or len(all_words_list[i]) >= 5:
            continue
        else:
            feature_words.append(all_words_list[i])

    return feature_words


# 根据 feature_words 将训练数据和测试数据向量化
def data_vector(feature_words: list, train_data_list: list, test_data_list: list) -> (list, list):
    """
    :param feature_words: 数据集的特征词汇表
    :param train_data_list: 训练数据，二维列表，每个元素表示一个新闻样本
    :param test_data_list: 测试数据，二维列表，每个元素表示一个新闻样本
    :return: 向量化的训练数据和测试数据
    """
    train_feature_list = []  # train_data_list 的向量化形式
    test_feature_list = []  # test_data_list 的向量化形式

    # 将训练数据向量化
    for sample in train_data_list:
        train_sample_list = []  # 用于存储训练集单个样本的特征词汇，元素个数与 feature_words 一致
        sample_set = set(sample)  # 将样本数据进行去重
        for word in feature_words:
            if word in sample_set:
                train_sample_list.append(1)
            else:
                train_sample_list.append(0)
        train_feature_list.append(train_sample_list)

    # 将测试数据向量化
    for sample in test_data_list:
        test_sample_list = []  # 用于存储测试集单个样本的特征词汇，元素个数与 feature_words 一致
        sample_set = set(sample)  # 将样本数据进行去重
        for word in feature_words:
            if word in sample_set:
                test_sample_list.append(1)
            else:
                test_sample_list.append(0)
        test_feature_list.append(test_sample_list)

    return train_feature_list, test_feature_list


if __name__ == '__main__':
    dir_path = r'D:\MachineLearning\SogouC\Sample'  # 数据集存放目录

    # 获取词汇表、训练数据、测试数据、训练标签、测试标签
    all_words_list, train_data_list, test_data_list, train_labels_list, test_labels_list = text_process(dir_path)

    # 生成 stopwords_set
    stopwords_file_path = r'D:\MachineLearning\stopwords_cn.txt'
    stopwords_set = stop_words_set(stopwords_file_path)

    # 获取数据集的特征词汇表
    feature_words = delete_words(all_words_list, stopwords_set)

    # 获取向量化的训练数据和测试数据
    train_feature_list, test_feature_list = data_vector(feature_words, train_data_list, test_data_list)
    # print(np.array(train_feature_list).shape)  # (73, 8747)
    # print(np.array(test_feature_list).shape)  # (17, 8747)

    # 实例化 MultinomialNB 对象
    clf = MultinomialNB()

    # 使用训练数据和训练标签进行拟合
    clf.fit(train_feature_list, train_labels_list)

    # 预测
    predict_result = clf.predict(test_feature_list)

    # 准确率
    accuracy = clf.score(test_feature_list, test_labels_list)

    print('测试结果为：', predict_result)
    print('准确率为：', accuracy)
