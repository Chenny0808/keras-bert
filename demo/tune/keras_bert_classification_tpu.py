#! user/bin/env python
# _*_ encoding: utf-8 _*_
"""
 @File Name: keras_bert_classification_tpu
 @Author:    Chenny
 @email:     15927299723@163.com
 @date:      2019/6/21 11:51
 @software:  PyCharm
"""

import os
import codecs
from keras_bert import load_trained_model_from_checkpoint
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from keras_bert import Tokenizer
from tensorflow.python import keras
from keras_bert import AdamWarmup, calc_train_steps
import tensorflow.keras.backend as K
from keras_bert import get_custom_objects

# 超参数
SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4

# @title Environment
pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'

# 加载bert词表和模型
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
    )

# 下载IMDB数据集
dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
    )

# 数据转为
tokenizer = Tokenizer(token_dict)
def load_data(path):
    global tokenizer
    indices, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r') as reader:
                text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments)
# 加载数据并转为词id
train_path = os.path.join(os.path.dirname(dataset), 'aclImdb', 'train')
test_path = os.path.join(os.path.dirname(dataset), 'aclImdb', 'test')
train_x, train_y = load_data(train_path)
test_x, test_y = load_data(test_path)

# 定义自定义模型
inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output  # 获取'NSP-Dense'层的输出
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)  # 稠密层 + softmax
decay_steps, warmup_steps = calc_train_steps(  # 指数衰减步数，热启动步数
    train_y.shape[0],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    )

model = keras.models.Model(inputs, outputs)
model.compile(  # 编译模型以供训练
    AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    )

# 初始化所有变量
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
    )
sess.run(init_op)

# 转为 tpu model
tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
strategy = tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    )
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

# 训练
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    tpu_model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

# 预测
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    predicts = tpu_model.predict(test_x, verbose=True).argmax(axis=-1)
