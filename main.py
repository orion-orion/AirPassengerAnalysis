'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-03 19:59:38
LastEditors: ZhangHongYu
LastEditTime: 2022-07-07 14:15:10
'''
from data_reader import load_data
from data_utils import change_data_format
from k_fold_cross_valid import k_fold_cross_valid, train_final
from predict import predict
import argparse

def parse_args():
    """parse the command line args

    Returns:
        args: a namespace object including args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        help="whether use the features selected or retrain the model to select the features"
        " possible are `load`,`retrain`",
        type=str,
        default='retrain'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
     
    # 加载数据
    df_train, df_test = load_data()

    # 将输入数据转换为特定的数据结构
    Xi_train, Xv_train, y_train, Xi_test, Xv_test = change_data_format(df_train, df_test)

    # k折交叉验证寻找最优超参数
    k_fold_cross_valid(Xi_train, Xv_train, y_train, mod=args.model)
    
    if args.model == 'retrain': #如果要求重新训练模型
        # 用所有数据来训练
        train_final(Xi_train, Xv_train, y_train)
    
    # 给出预测结果
    predict(Xi_test, Xv_test, df_test)

