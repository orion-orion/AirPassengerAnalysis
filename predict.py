'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-06 11:03:23
LastEditors: ZhangHongYu
LastEditTime: 2022-07-06 17:05:30
'''
import config
from pnn import PNN
import tensorflow as tf
import pandas as pd
import os
import numpy as np

def predict(Xi_test, Xv_test, df_test: pd.DataFrame):
    model_save_dir = os.path.join(config.model_root, "final")
    if not os.path.exists(model_save_dir):
        raise IOError(f"Cant find the path {model_save_dir}")
    if not os.path.exists(config.predicted_root):
       os.makedirs(config.predicted_root)
    
     
    predict_iter = tf.data.Dataset.from_tensor_slices((Xi_test, Xv_test)).batch(config.predict_batch_size)   
    pnn = PNN(**config.pnn_params)
    pnn = tf.keras.models.load_model(model_save_dir)
    # predict 需要关闭dropout
    pnn.dropout_deep = [0] * len(pnn.dropout_deep)

    y_hat_list, y_hat_category_list = [], []
    for _, (Xi, Xv) in enumerate(predict_iter):
        y_hat = pnn([Xi ,Xv])
        y_hat_category = tf.cast(tf.logical_not(tf.less(y_hat, [0.5])), tf.int32)        
        y_hat_list.append(y_hat)
        y_hat_category_list.append(y_hat_category)
        
    y_hat = np.concatenate(y_hat_list, axis=0)
    y_hat_category = np.concatenate(y_hat_category_list, axis=0)

    df_test["y_hat"], df_test["emd_lable2"] = y_hat, y_hat_category
    df_test = df_test[df_test["emd_lable2"]==1]
    df_test.sort_values(by=["y_hat"], ascending=True)
    if df_test.shape[0] > config.max_passenger:
        df_test = df_test.iloc[:config.max_passenger, :]
    df_test = df_test.drop("y_hat", axis=1)
    df_test.to_csv(os.path.join(config.predicted_root, "predicted_passenger.csv"))
    
    
