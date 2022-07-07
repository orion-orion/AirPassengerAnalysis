'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-06 10:14:40
LastEditors: ZhangHongYu
LastEditTime: 2022-07-07 11:20:04
'''
import pandas as pd
import config

class FeatureDictionary(object):
    def __init__(self,trainfile=None,testfile=None,
                 df_train=None,df_test=None,numeric_cols=[],
                 ignore_cols=[]):
        assert not ((trainfile is None) and (df_train is None)), "trainfile or df_train at least one is set"
        assert not ((trainfile is not None) and (df_train is not None)), "only one can be set"
        assert not ((testfile is None) and (df_test is None)), "testfile or df_test at least one is set"
        assert not ((testfile is not None) and (df_test is not None)), "only one can be set"

        self.trainfile = trainfile
        self.testfile = testfile
        self.df_train = df_train
        self.df_test = df_test
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.df_train is None:
            df_train = pd.read_csv(self.trainfile)
        else:
            df_train = self.df_train
        if self.df_test is None:
            df_test = pd.read_csv(self.testfile)
        else:
            df_test = self.df_test
        df = pd.concat([df_train,df_test])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                # 后面是one-hot对应维度的列索引
                # [10 11  7  6  9  5  4  8  3  0  2  1 -1]
                # {10: 79, 11: 80, 7: 81, 6: 82, 9: 83, 5: 84, 4: 85, 8: 86, 3: 87, 0: 88, 2: 89, 1: 90, -1: 91}
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self,infile=None,df=None):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        ye = dfi['emd_lable2'].values.tolist()
        dfi.drop(['emd_lable2'],axis=1,inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col,axis=1,inplace=True)
                dfv.drop(col,axis=1,inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()
        return xi, xv, ye
    
def change_data_format(df_train, df_test):
    numeric_cols = []
    ignore_cols = []
    for col in df_train.columns:
        type_col = str(df_train[col].dtype)
        if type_col == 'float64' or type_col == 'int64':
            numeric_cols.append(col)

    fd = FeatureDictionary(df_train=df_train,
                           df_test=df_test,
                           numeric_cols=numeric_cols,
                           ignore_cols=ignore_cols)
    data_parser = DataParser(feat_dict= fd)

    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    # 这里不方便调用imblearn实现过采样，因为他不是直接存储为one-hot矩阵
    # 而是索引和值分开存储的。也就是说，要用tensorflow自带的embedding函数，
    # 就很难再调用imlearn中的过采样了
    Xi_train, Xv_train, y_train = data_parser.parse(df=df_train)
    Xi_test, Xv_test, _ = data_parser.parse(df=df_test)

    # Xv_test中有异常数据需要专门过滤掉
    Xv_test = [[ 0 if isinstance(x, str) else x for x in row] for row in Xv_test] 

    config.pnn_params['feature_size'] = fd.feat_dim # 包括one-hot所有维度的总维度,n_all_feature
    config.pnn_params['field_size'] = len(Xi_train[0]) # 将one-hot看做整体的总的域个数, n_field

    return  Xi_train, Xv_train, y_train, Xi_test, Xv_test