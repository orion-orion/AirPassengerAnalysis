'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-01 11:33:19
LastEditors: ZhangHongYu
LastEditTime: 2022-07-06 10:41:13
'''
import pandas as pd
import config


def data_preprocess(data):
    #  获得每个特征的缺失信息
    null_info = data.isnull().sum(axis=0)
    #  丢弃缺失值多于30%的特征
    features = [k for k, v in dict(
        null_info).items() if v < data.shape[0] * 0.3]
    data = data[features]

    null_info = data.isnull().sum(axis=0)
    # 选去出需要填补缺失值的特征
    features_fillna = [k for k, v in dict(null_info).items() if v > 0]

    # 对缺失值进行填补
    for feature in features_fillna:
        # 如果是非数值型特征或者是整型离散数值，用众数填补
        # 将列按出现频率由高到低排序，众数即第一行，inplace表示原地修改
        if str(data[feature].dtype) == 'object' or str(data[feature].dtype) == 'int64':
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mode().iloc[0]
            )
        # 浮点连续数值型特征插值填补+平均数处理边缘
        else:
            # 先将中间的数据插值处理
            data.loc[:,  feature] = data[feature].interpolate(
                method="zero", axis=0, limit_direction='both')
            # 边缘直接填充平均数
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mean()
            )
    return data


def load_data():
    df_train = pd.read_csv(config.train_file)
    df_test = pd.read_csv(config.test_file)
    df_train = data_preprocess(df_train)
    df_test = data_preprocess(df_test)
    return df_train, df_test
