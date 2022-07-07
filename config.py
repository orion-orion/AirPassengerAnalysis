'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-02 18:52:47
LastEditors: ZhangHongYu
LastEditTime: 2022-07-07 11:08:36
'''
import tensorflow as tf

train_file = "data/train.csv"
test_file = "data/test.csv"
num_splits = 3  # 交叉验证折数
random_seed = 2020
predicted_root = "prediction"
max_passenger = 500
model_root = "model"

pnn_params = {
    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "verbose": True,
    "random_seed": random_seed,
    "loss_type": "binary_crossentropy",
    "deep_init_size": 50,
    "use_inner": False
}

train_params = {
    "loss_type": "binary_crossentropy",
    "learning_rate": 0.01,
    "epochs": 1,
    "optimizer_type": "adam",
    "batch_size": 4
}

predict_batch_size = 4
