'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-06 10:11:25
LastEditors: ZhangHongYu
LastEditTime: 2022-07-07 14:56:45
'''
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
import config
from sklearn.model_selection import StratifiedKFold
from pnn import PNN


def evaluate_accuracy(data_iter, net):
    acc_sum , n = 0.0, 0
    f1_y, f1_y_hat, roc_auc_y, roc_auc_y_hat = [], [], [], [] # 样本不均衡可能导致在该折交叉验证中没有正例，计算不了auc
    # evaluate 需要关闭dropout
    net.dropout_deep = [0] * len(net.dropout_deep)
    for _, (Xi, Xv, y) in enumerate(data_iter):
        y_hat = net([Xi ,Xv])
        
        # 先用输出的概率计算auc
        roc_auc_y.append(y.numpy().reshape(-1, 1))
        roc_auc_y_hat.append(y_hat.numpy())
        
        y_hat = tf.cast(tf.logical_not(tf.less(y_hat, [0.5])), tf.int32)
        #此题输出y_hat就是p+无用，一般情况下对所有测试集样本的 I(argmax(p_label1, p_label2,...) == yi) 进行求和
        acc_sum += tf.reduce_sum(
            tf.cast(
                tf.equal(
                    # dropout_deep必须要在[0,1)之间
                    tf.reshape(y_hat, shape=(-1,)),
                    y
                ),
                dtype=tf.int32
            )
        ).numpy()

        f1_y.append(y.numpy().reshape(-1, 1))
        f1_y_hat.append(y_hat.numpy())

        n += y.shape[0]

    valid_f1 =  f1_score(np.concatenate(tuple(f1_y), axis=0), np.concatenate(tuple(f1_y_hat), axis=0))
    # 样本不均衡，很可能没有正例，故这里计算auc很有可能出错
    valid_roc_auc = roc_auc_score(np.concatenate(tuple(roc_auc_y), axis=0), np.concatenate(tuple(roc_auc_y_hat), axis=0))
    return acc_sum / n, valid_f1, valid_roc_auc


def train(net, train_iter, fold, loss_type, learning_rate, epochs, optimizer_type, batch_size):
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        f1_y, f1_y_hat, roc_auc_y, roc_auc_y_hat = [], [], [], [] # 样本不均衡可能导致在该折交叉验证中没有正例，计算不了auc
        # 所有样本的loss,　所有样本中预测正确的个数，所有样本总个数
        for Xi, Xv, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net([Xi, Xv]) 
                # loss
                if loss_type == 'binary_crossentropy':
                    loss = tf.keras.losses.binary_crossentropy(y, y_hat) 
                elif loss_type == "mse":
                    loss = tf.keras.losses.MSE(y, y_hat)
                # 返回的是这个batch的loss向量，需要对其求平均
                loss = tf.reduce_mean(loss) 
            # unconnected_gradients为无法求导时返回的值，有none和zero可选择，默认为none
            # 这里建议用zero，否则后面grad/batch_size要报错
            grads = tape.gradient(loss, net.trainable_variables, unconnected_gradients="zero") 

            # optimizer
            if optimizer_type == 'sgd':
                optimizer = tf.keras.optimizers.SGD(learning_rate)
            elif optimizer_type == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                                    epsilon=1e-8)
            elif optimizer_type == "adagrad":
                optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate,
                                                            initial_accumulator_value=1e-8)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            train_l_sum += loss.numpy()

            # 先用输出的概率计算auc
            roc_auc_y.append(y.numpy().reshape(-1, 1))
            roc_auc_y_hat.append(y_hat.numpy())

            # 以多分类为例，计算损失函数时用概率输出P(Y=0,1,2...|x;theta)组成的向量与y独热向量求交叉熵即可
            # 但算精度要离散化，要对一个batch的I(argmax(p_label1, p_label2,...) == yi)进行求和 
            # 二分类 >=0.5 预测为正类, <0.5　预测为负类
            y_hat = tf.cast(tf.logical_not(tf.less(y_hat, [0.5])), tf.int32)
            train_acc_sum += tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.reshape(y_hat, shape=(-1,)),
                        y
                    ), 
                    dtype=tf.int32
                )
            ).numpy() # 这里是二分类直接输出正例概率值, 一般情况下多分类对一个batch的 I(argmax(p_label1, p_label2,...) == yi) 进行求和

            f1_y.append(y.numpy().reshape(-1, 1))
            f1_y_hat.append(y_hat.numpy())
            
            n += y.shape[0] # n是总样本个数，这里累加n_batch
        # 训练集的相关评估指标
        train_loss = train_l_sum/n
        train_acc = train_acc_sum/n

        train_f1 =  f1_score(np.concatenate(tuple(f1_y), axis=0), np.concatenate(tuple(f1_y_hat), axis=0))
        # 样本不均衡，很可能没有正例，故这里计算auc很有可能出错
        train_roc_auc = roc_auc_score(np.concatenate(tuple(roc_auc_y), axis=0), np.concatenate(tuple(roc_auc_y_hat), axis=0))

        print('fold %d, epoch %d, loss %.4f, train acc %.3f, train f1 %.3f, train auc %.3f' 
        % (fold + 1, epoch + 1, train_loss, train_acc, train_f1, train_roc_auc))


def k_fold_cross_valid(Xi_train, Xv_train, y_train, mod):

    # folds
    folds = list(StratifiedKFold(n_splits=config.num_splits, shuffle=True,
                                 random_state=config.random_seed).split(Xi_train, y_train))

    _get = lambda x,l:[x[i] for i in l]

    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
            
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        pnn = PNN(**config.pnn_params)
        
        train_iter = tf.data.Dataset.from_tensor_slices((Xi_train_, Xv_train_, y_train_)).batch(config.train_params['batch_size'])
        valid_iter = tf.data.Dataset.from_tensor_slices((Xi_valid_, Xv_valid_, y_valid_)).batch(config.train_params['batch_size'])

        kfold_model_dir = os.path.join(config.model_root, f"{i + 1}th_fold")
        if mod == "retrain":
            if not os.path.exists(kfold_model_dir):
                os.makedirs(kfold_model_dir)
            train(
                pnn, train_iter, i, **config.train_params)
            pnn.save(kfold_model_dir) 
        
        
        if not os.path.exists(kfold_model_dir):
            raise IOError(f"Cant find the path {kfold_model_dir}")
        pnn = tf.keras.models.load_model(kfold_model_dir)
            
        valid_acc, valid_f1, valid_roc_auc = evaluate_accuracy(valid_iter, pnn)
        print('fold %d, valid acc %.3f, valid f1 %.3f, valid auc %.3f' 
        % (i + 1, valid_acc, valid_f1, valid_roc_auc))



def train_final(Xi_train, Xv_train, y_train):
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
        
    pnn = PNN(**config.pnn_params)
    train_iter = tf.data.Dataset.from_tensor_slices((Xi_train, Xv_train, y_train)).batch(config.train_params['batch_size'])
    train(
        pnn, train_iter, None, **config.train_params)

    final_model_dir = os.path.join(config.model_root, "final")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    pnn.save(final_model_dir) 

        