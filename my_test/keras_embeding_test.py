# %%

# 运行参数： --model_dir=E:\model\official_myDataset --data_dir=E:\data\my_dataset --train_epochs=10 --distribution_strategy=one_device --num_gpus=1 --download

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a simple model on the MNIST dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils

# from official.utils.misc import model_helpers
# from official.vision.image_classification.resnet import common
# from sklearn.metrics import f1_score, precision_recall_fscore_support
# import numpy as np
# import json

FLAGS = flags.FLAGS

feat_to_onehot = [[3, 5, 10, 5], [4, 6, 4, 8]]

from tensorflow.keras import layers
# layers.Embedding()()
# %%
def build_model(desc_list_tuple):
    # desc_list_tuple描述了输入有几个特征，他们分别有多少特征值，每个特征值映射到几维向量。
    # 其中左边数组表示输入尺度，右侧数组表示映射后尺度
    # 如[[3,5,10,5],[4,6,4,8]] 表示一共有四种特征，第一种特征有三种取值，分别映射成一个4维向量（参数数量为3*4）,
    # input的第一维就是[1,0,0]或[0,1,0]或[0,0,1]，表示-1,0,1三种取值

    input = tf.keras.layers.Input(shape=(sum(desc_list_tuple[0]),), name='input')

    xs = tf.split(input,num_or_size_splits=desc_list_tuple[0],axis=-1)
    embedings = []
    # Customized_softplus = tf.keras.layers.Lambda(lambda x: tf.nn.softplus(x))
    for i, x in enumerate(xs):
        embedings.append(
            tf.keras.layers.Dense(desc_list_tuple[1][i], activation=None, use_bias=False)(x)
        )
    y = tf.keras.layers.Concatenate()(embedings)

    probs = tf.keras.layers.Dense(2, activation='softmax')(y)

    model = tf.keras.models.Model(input, probs, name='my_emb_test')

    return model


# %%
def run(flags_obj, datasets_override=None, strategy_override=None):
    strategy = strategy_override or distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus
    )

    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    # dsc = tfds.builder_cls('my_recom_dataset')
    # rr = dsc(data_dir=flags_obj.data_dir)

    ds = tfds.builder('my_recom_dataset', data_dir=flags_obj.data_dir)

    if flags_obj.download:
        ds.download_and_prepare()

    mnist_train, mnist_vali = datasets_override or ds.as_dataset(
        split=['train', 'test'],

    )

    train_input_dataset = mnist_train.cache().repeat(10000).shuffle(buffer_size=100000).batch(32)
    eval_input_dataset = mnist_vali.cache().repeat().batch(300)

    with strategy_scope:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.002, decay_steps=100000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model = build_model(feat_to_onehot)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

    for indx, (train_data, vali_data) in enumerate(zip(train_input_dataset.as_numpy_iterator(), eval_input_dataset.as_numpy_iterator())):
        print(model.train_on_batch(
            x=train_data['input'],
            y=train_data['label'],
            class_weight={0: 1, 1: 1}
        ))

    export_path = os.path.join(flags_obj.model_dir, 'saved_model')
    model.save(export_path, include_optimizer=False)

    return 'over'

# tf.io.gfile.GFile().readlines()
# %%
def define_mnist_flags():
    """Define command line flags for MNIST model."""
    flags_core.define_base(
        clean=True,
        num_gpu=True,
        train_epochs=True,
        epochs_between_evals=True,
        distribution_strategy=True)

    flags_core.define_device()
    flags_core.define_distribution()

    # flags.DEFINE_string('data_dir', r'D:\tf2_official_data_model_res\data\my_recommand_data','null')
    # flags.DEFINE_string('model_dir', r'D:\tf2_official_data_model_res\model\my_recommand_model', 'null')
    # flags.DEFINE_string('distribution_strategy', r'one_device', 'null')
    FLAGS.set_default('data_dir',  r'D:\tf2_official_data_model_res\data\my_recommand_data')
    FLAGS.set_default('model_dir', r'D:\tf2_official_data_model_res\model\my_recommand_model')
    FLAGS.set_default('distribution_strategy', r'one_device')
    FLAGS.set_default('num_gpus',0)
    FLAGS.set_default('train_epochs', 100)
    flags.DEFINE_bool('download', True,'Whether to download data to `--data_dir`.')


# %%
def main(_):
    stats = run(flags.FLAGS)
    logging.info('Run stats:\n%s', stats)


# %%

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_mnist_flags()
    app.run(main)

