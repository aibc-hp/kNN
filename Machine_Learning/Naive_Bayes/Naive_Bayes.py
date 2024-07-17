# -*- coding: utf-8 -*-
# @Time    : 2023/12/1 11:01
# @Author  : aibc-hp
# @File    : Naive_Bayes.py
# @Project : OXI_Model_Variant_CNN-OPTIM_clean-data_VGG_Normalize
# @Software: PyCharm

import numpy as np
from functools import reduce


# 读取数据
def read_dataset() -> (list, list):
    """
    :return: 返回样本数据集和样本标签
    """
    # 将留言进行单词切分，并转换成词向量
    samples = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 各样本对应标签，1 代表内容不当，2 代表内容得当
    labels = [2, 1, 2, 1, 2, 1]

    return samples, labels


# 依据数据集创建词汇表
def create_vocabulary(dataset: list) -> list:
    """
    :param dataset: 样本数据集
    :return: 数据集中出现的所有词汇集合，以列表形式返回
    """
    vocab_set = set([])  # 创建一个空集
    for sample in dataset:
        vocab_set = vocab_set | set(sample)  # 取并集

    return list(vocab_set)


# 用词汇表的稀疏向量形式来表示每个词向量样本
def vocab_vector_to_vocabulary_vector(vocabulary: list, sample: list) -> list:
    """
    :param vocabulary: 词汇表
    :param sample: 样本数据集中的一个样本
    :return: 词汇表的稀疏向量
    """
    vocabulary_vector = [0] * len(vocabulary)  # 元素个数与词汇表 vocabulary 一致
    for vocab in sample:
        if vocab in vocabulary:  # 如果该词汇出现在词汇表中，则将 vocabulary_vector 对应位置的值置为 1
            vocabulary_vector[vocabulary.index(vocab)] = 1
        else:
            print(f"{vocab} is not in vocabulary!")

    return vocabulary_vector


# 训练朴素贝叶斯分类器
def train_naive_bayes_classifier(train_mat: list, train_labels: list) -> (np.ndarray, np.ndarray, float):
    """
    :param train_mat: 训练样本数据，都已转成词汇表的稀疏向量形式
    :param train_labels: 训练样本数据的对应标签
    :return: 返回在类别 1 和 2 情况下各个词汇出现的概率以及类别 1 占样本集的概率
    """
    num_samples = len(train_mat)  # 训练样本数；6
    num_vocabs = len(train_mat[0])  # 每个样本向量含有多少个元素；32

    p_1 = float(sum([1 for label in train_labels if label == 1]) / num_samples)  # 在训练集中，内容不当的概率
    p_1_vocab = np.zeros(num_vocabs)  # 当样本标签为 1 时，每个词出现的次数
    p_2_vocab = np.zeros(num_vocabs)  # 当样本标签为 2 时，每个词出现的次数
    p_1_vocabs = 0.0  # 当样本标签为 1 时，所有词出现的次数和
    p_2_vocabs = 0.0  # 当样本标签为 2 时，所有词出现的次数和

    for i in range(num_samples):
        if train_labels[i] == 1:
            p_1_vocab += train_mat[i]
            p_1_vocabs += sum(train_mat[i])
        else:
            p_2_vocab += train_mat[i]
            p_2_vocabs += sum(train_mat[i])

    p_1_vector = p_1_vocab / p_1_vocabs  # 在类别 1 的情况下，各个词出现的概率；P(w1|1)、P(w2|1)、P(w3|1)、...
    p_2_vector = p_2_vocab / p_2_vocabs  # 在类别 2 的情况下，各个词出现的概率；P(w1|2)、P(w2|2)、P(w3|2)、...

    return p_1_vector, p_2_vector, p_1


# 预测分类结果
def predict(predict_data: np.ndarray, p_1_vector: np.ndarray, p_2_vector: np.ndarray, p_1: float) -> int:
    """
    :param predict_data: 预测数据，已转成词汇表的稀疏向量形式
    :param p_1_vector: 在类别 1 的情况下，各个词出现的条件概率
    :param p_2_vector: 在类别 2 的情况下，各个词出现的条件概率
    :param p_1: 类别为 1 的先验概率
    :return: 预测的类别
    """
    p1 = reduce(lambda x, y: x * y, predict_data * p_1_vector) * p_1  # 类别为 1 的概率
    p2 = reduce(lambda x, y: x * y, predict_data * p_2_vector) * (1 - p_1)  # 类别为 2 的概率

    print('p1:', p1)
    print('p2:', p2)

    if p1 > p2:
        return 1
    else:
        return 2


if __name__ == '__main__':
    # 获取样本数据和对应标签
    samples, labels = read_dataset()

    # 获取词汇表
    vocabulary = create_vocabulary(samples)

    # 获取训练的样本数值向量
    train_mat = []
    for sample in samples:
        train_mat.append(vocab_vector_to_vocabulary_vector(vocabulary, sample))

    # 获取在类别 1 和 2 情况下各个词汇出现的概率以及类别 1 占样本集的概率
    # p_1_vector 中存放的是各个单词在类别 1 情况下出现的条件概率
    # p_2_vector 中存放的是各个单词在类别 2 情况下出现的条件概率
    # p_1 就是类别为 1 的先验概率
    p_1_vector, p_2_vector, p_1 = train_naive_bayes_classifier(train_mat, labels)

    # 测试
    predict_data = ['love', 'my', 'dalmation']  # 预测样本
    predict_data_vector = np.array(vocab_vector_to_vocabulary_vector(vocabulary, predict_data))  # 将预测样本转成词汇表的稀疏向量
    result = predict(predict_data_vector, p_1_vector, p_2_vector, p_1)

    if result == 1:
        print(f'{predict_data} 属于内容不当')
    else:
        print(f'{predict_data} 属于内容得当')


"""
P(1|X)P(X)=P(X|1)P(1) => P(1|(w1, w2, ..., w32))P(w1, w2, ..., w32)=P((w1, w2, ..., w32)|1)P(1)=P(w1|1)P(w2|1)···P(w32|1)P(1)
P(2|X)P(X)=P(X|2)P(2) => P(2|(w1, w2, ..., w32))P(w1, w2, ..., w32)=P((w1, w2, ..., w32)|2)P(2)=P(w1|2)P(w2|2)···P(w32|2)P(2)
比较 P(1|X) 与 P(2|X) 的大小，将更大值所属类别作为最终的分类结果
"""