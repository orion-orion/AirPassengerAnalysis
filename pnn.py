'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-03 14:51:59
LastEditors: ZhangHongYu
LastEditTime: 2022-07-07 10:57:53
'''
import numpy as np
import tensorflow as tf


class PNN(tf.keras.Model):

    def __init__(self, feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size=50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="binary_crossentropy",
                 use_inner=True):
        super().__init__()
        assert loss_type in ["binary_crossentropy", "mse"], \
            "loss_type can be either 'binary_crossentropy' for classification task or 'mse' for regression task"

        self.feature_size = feature_size    # M one-hot编码后的所有特征数，即类别型特征的取值范围（考虑所有类别型特征）
        self.field_size = field_size    # F one-hot编码前的所有特征域数量
        self.embedding_size = embedding_size    # K 嵌入后单个特征域的向量维度

        self.deep_layers = deep_layers       # 全连接层大小
        self.deep_init_size = deep_init_size  # Product Layer大小
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type

        self.use_inner = use_inner

        # init其他变量
        tf.random.set_seed(self.random_seed)

        # 这里的weights是一个字典，包括所有weight和bias
        self._weights = self._initialize_weights()

    def _initialize_weights(self):
        weights = dict()
        # embeddings
        weights['feature_embeddings'] = tf.Variable(tf.random.normal([self.feature_size,
                                                                      self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random.normal([self.feature_size,
                                                                1], 0.0, 1.0), name='feature_bias')

        # Product Layers
        # 线性部分
        weights['product-linear'] = tf.Variable(tf.random.normal([self.deep_init_size,
                                                                  self.field_size, self.embedding_size], 0.0, 0.01))
        weights['product-bias'] = tf.Variable(tf.random.normal([self.deep_init_size, ],
                                                               0.0, 1.0))
        # 非线性部分
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(tf.random.normal([self.deep_init_size,
                                                                               self.field_size], 0.0, 0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random.normal([self.deep_init_size, self.embedding_size, self.embedding_size],
                                 0.0, 0.01)
            )

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.deep_init_size
        # 先初始化第0个全连接层
        # glorot 权值初始化方法，U~( -sqrt(6/(din+dout)), +sqrt(6/(din + dout)))
        glorot = np.sqrt(6.0/(input_size + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(tf.random.uniform(
            [input_size, self.deep_layers[0]], -glorot, glorot))
        weights['bias_0'] = tf.Variable(tf.random.uniform(
            [1, self.deep_layers[0]], -glorot, glorot))
        # 依次递推地初始化最后一层之前的全连接层
        for i in range(1, num_layer):
            glorot = np.sqrt(
                6.0/(self.deep_layers[i - 1] + self.deep_layers[i]))
            weights['layer_%d' % i] = tf.Variable(tf.random.uniform([self.deep_layers[i-1],
                                                                     self.deep_layers[i]], -glorot, glorot))
            weights['bias_%d' % i] = tf.Variable(
                tf.random.uniform([1, self.deep_layers[i]], -glorot, glorot))
        weights['batch_norm_scale_0'] = tf.Variable(
            np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)
        weights['batch_norm_offset_0'] = tf.Variable(
            np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)

        # 初始化最后一层全连接层
        glorot = np.sqrt(5.0/(input_size + 1))
        weights['output'] = tf.Variable(tf.random.uniform(
            [self.deep_layers[-1], 1], -glorot, glorot))
        weights['output_bias'] = tf.Variable(
            tf.constant(0.01), dtype=np.float32)
        return weights

    def call(self, features):
        """
        关于Embeddings：
        self._weights['feature_embeddings]为(feature_size, embedding_size),(256 * 8),其中feature_size是把所有one-hot展开后的总维度
        而feat_index为=(batch_size, n_field=38，1), n_field为one-hot没展开时的域个数
        根据索引从W里面选n_field个被嵌入后的向量出来，最终得到(batch_size, n_field, 8)，38只是元素个数，实际上字典大小即元素取值范围为256

        feat_index是embedding层的输入，也是对embedding矩阵的索引，比如输入index为3元素列表 ，也可以对W=(8, 2)进行索引
        其实就是从(8, 2)里面选3个出来二维向量出来，最终得到(3, 2)，feat_index每一个位置就是一个索引
        """
        feat_index, feat_value = features[0], features[1]

        # 所有特征域都进行嵌入，数值域也当做类型域进行处理
        # 由于所有样本的某个数值域的索引都相同，故嵌入到相同的向量
        embeddings = tf.nn.embedding_lookup(
            self._weights['feature_embeddings'], feat_index)
        # (batch_size, n_field, 1)
        feat_value = tf.reshape(feat_value, shape=[-1, self.field_size, 1])
        # 逐个元素相乘，恢复数值域的信息
        # 因为类型域对应的feat_value都是1，故不受影响
        # None * F * K (F为特征域数量)
        embeddings = tf.multiply(embeddings, feat_value)

        # 乘积层
        # 线性部分
        linear_output = []
        for i in range(self.deep_init_size):
            linear_output.append(tf.reshape(tf.reduce_sum(tf.multiply(embeddings, \
                self._weights['product-linear'][i]), axis=[1, 2]), shape=(-1, 1)))
        lz = tf.concat(linear_output, axis=1)  # None * init_deep_size

        # 非线性部分
        quadratic_output = []
        if self.use_inner:  # 使用内积
            for i in range(self.deep_init_size):
                delta = tf.multiply(embeddings, tf.reshape(
                    self._weights['product-quadratic-inner'][i], (1, -1, 1)))  # None * F * K
                quadratic_output.append(tf.reshape(tf.norm(tf.reduce_sum(
                    delta, axis=1), axis=1), shape=(-1, 1)))   # None * 1
        else:  # 使用外积
            embedding_sum = tf.reduce_sum(embeddings, axis=1)
            p = tf.matmul(tf.expand_dims(embedding_sum, 2),
                          tf.expand_dims(embedding_sum, 1))  # None * K * K
            for i in range(self.deep_init_size):
                delta = tf.multiply(p, tf.expand_dims(
                    self._weights['product-quadratic-outer'][i], 0))  # None * K * K
                quadratic_output.append(tf.reshape(tf.reduce_sum(
                    delta, axis=[1, 2]), shape=(-1, 1)))   # None * 1
        lp = tf.concat(quadratic_output, axis=1)  # None * init_deep_size

        # L1 deep component
        y_deep = tf.nn.relu(lz + lp + self._weights['product-bias'])
        y_deep = tf.nn.dropout(y_deep, self.dropout_deep[0])
 
        # L2 deep component
        for i in range(0, len(self.deep_layers)):
            y_deep = tf.matmul(
                y_deep, self._weights["layer_%d" % i]) + self._weights["bias_%d" % i]
            mean, variance = tf.nn.moments(y_deep, axes=0)
            y_deep = tf.nn.batch_normalization(
                y_deep, mean, variance,
                self._weights['batch_norm_scale_0'],
                self._weights['batch_norm_offset_0'],
                variance_epsilon=1e-10
            )
            y_deep = self.deep_layers_activation(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_deep[i + 1])

        # output ctr
        out = tf.matmul(
            y_deep, self._weights['output']) + self._weights['output_bias']
        # binary_crossentropy为交叉熵的二类别情况 yp +(1-y)(1-p) y=0,1
        if self.loss_type == 'binary_crossentropy':
            # 二分类，此处用sigmoid，多分类是softmax
            out = tf.nn.sigmoid(out)
        return out
