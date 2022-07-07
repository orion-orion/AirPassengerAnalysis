<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2021-09-19 19:53:49
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-07 15:37:20
-->

# 航空旅客付费选座意愿识别

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/AirPassengerAnalysis)
[![](https://img.shields.io/github/license/orion-orion/AirPassengerAnalysis)](https://github.com/orion-orion/AirPassengerAnalysis)
[![](https://img.shields.io/github/stars/orion-orion/AirPassengerAnalysis?style=social)](https://github.com/orion-orion/AirPassengerAnalysis)
[![](https://img.shields.io/github/issues/orion-orion/AirPassengerAnalysis)](https://github.com/orion-orion/AirPassengerAnalysis)

## 赛题要求

本项目为2021年软件服务外包大赛《航空旅客付费选座意愿识别》赛题，该赛题要求从650个特征因子中，提取或衍生出能够描述旅客付费选座意愿的特征因子集合；并利用特征因子集合的成果，构建旅客付费选座意愿识别模型。

## 模型架构

由于题目给定的特征非常多，我们采用PNN(Product-based-Neural-Network)深度学习模型对特征工程进行模型化。模型原始论文在项目目录`data`下。该模型的Embedding层能够对类别型特征组成的稀疏矩阵转换为大量稠密矩阵，提高模型收敛速率，同时Product层能够将特征域的Embedding后的向量进行两两交叉，充分挖掘组合特征，获取特征之间的交叉信息。

## 环境依赖

运行以下命令安装环境依赖：

```
pip install -r requirements.txt
```

## 数据集

数据集直接采用的赛方给定的旅客付费选座数据集，放在项目目录中的 `data`文件夹下。



## 数据格式转换
项目在对模型进行训练前会先将数据转换为以下格式：
- [ ] **Xi**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* 是数据集中样本 *i* 的特征域 *j* 的特征索引
- [ ] **Xv**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* 是数据集中样本 *i* 的特征域 *j* 的特征数值
    - *vali_j* 可以是二值型（1/0，用于二值/类别型特征）或浮点数（例如，10.24，用于数值特征）
- [ ] **y**: 数据集中每个样本的预测目标（1/0用于分类，数值用于回归）

请参阅`data_reader.py`查看如何将数据转换为项目所需格式。



## 模型

我本地训练好的模型（包括用于K-fold交叉验证的模型和最终训练的模型）都已经保存在项目的`model`目录下，可直接进行测试。

## 项目目录说明

-data  -------------------  存放数据

-model  -------------------  存放相关模型（包括用于K-fold交叉验证的模型和最终训练的模型）

-prediction  -------------------  存放对有付费选座意愿旅客的预测结果

-config.py  -------------------  存储模型的超参数，包括架构超参数、训练超参数等

-data_reader.py  -------------------  数据的读取和预处理的实现

-data_utils.py   -------------------  对输入数据格式进行转换的实现（处理各特征域得到 `Xi`和 `Xv`）

-k_fold_cross_valid.py  -------------------  在训练集和验证集上进行K-fold交叉验证逻辑的实现

-main.py  -------------------  主文件，用于从构建特征工程到模型的训练与评估的pipline

-pnn.py  -------------------  PNN模型架构的实现

-predict.py  -------------------  调用训练好的PNN模型对有付费选座意愿的旅客进行预测


## 使用方法

运行:

```
python main.py --model retrain
```

`model`参数表示是否重新开始训练PNN模型，若需重新训练模型可将 `model `参数设置为 `retrain `，否则设置为 `load`直接加载已经训练并保存好的模型（但前提是模型已经放置于 `model`目录下）。
