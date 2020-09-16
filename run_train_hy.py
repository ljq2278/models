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




################################################################################################

# 离线任务运行要做的准备工作


# import os

# # 相关包的安装

# f = os.popen('python -m pip install --upgrade pip')
# print(f.readlines())
# f = os.popen('pip install tensorflow-datasets==3.2.1')
# print(f.readlines())

# f = os.popen('pip install /code/tf_official_codes/research/pycocotools-2.0.1.tar.gz')
# print(f.readlines())
# f = os.popen('cd /code/tf_official_codes/official&&pip install -r requirements.txt')
# print(f.readlines())
# f = os.popen('pip install Cython==0.27.3')
# print(f.readlines())
# f = os.popen('pip install protobuf==3.12.2')
# print(f.readlines())

import tensorflow as tf

# rfds自定义数据集添加与gcs(谷歌云存储)功能关闭
import os
package_rootpath = tf.__file__.split('tensorflow/__init__.py')[0]
f = os.popen('cp my_recom_dataset.py ' + package_rootpath + 'tensorflow_datasets/structured/')
print(f.readlines())
f = os.popen('cp my_hy_dataset_multi_process.py ' + package_rootpath + 'tensorflow_datasets/structured/')
print(f.readlines())
f = os.popen('cp my_hy_dataset.py ' + package_rootpath + 'tensorflow_datasets/structured/')
print(f.readlines())
f = os.popen('cp __init__.py ' + package_rootpath + 'tensorflow_datasets/structured/')
print(f.readlines())
f = os.popen('cp gcs_utils.py ' + package_rootpath + 'tensorflow_datasets/core/utils/')
print(f.readlines())


#################################################################################################

import os
import numpy as np
from sklearn import metrics

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


# %%
def build_model():

    keywords = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,name='keywords')
    layer_words_emb = tf.keras.layers.Embedding(3966+1, 32)
    keywords_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_words_emb(keywords))

    disp_cate1 = tf.keras.layers.Input(shape=(1,), name='disp_cate1')
    layer_disp_cate1_emb = tf.keras.layers.Embedding(20, 32)
    disp_cate1_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_disp_cate1_emb(disp_cate1))

    disp_cate2 = tf.keras.layers.Input(shape=(1,), name='disp_cate2')
    layer_disp_cate2_emb = tf.keras.layers.Embedding(176, 32)
    disp_cate2_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_disp_cate2_emb(disp_cate2))

    disp_local1 = tf.keras.layers.Input(shape=(1,), name='disp_local1')
    layer_disp_local1_emb = tf.keras.layers.Embedding(666, 32)
    disp_local1_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_disp_local1_emb(disp_local1))

    disp_local2 = tf.keras.layers.Input(shape=(1,), name='disp_local2')
    layer_disp_local2_emb = tf.keras.layers.Embedding(3431, 32)
    disp_local2_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_disp_local2_emb(disp_local2))

    slot_id = tf.keras.layers.Input(shape=(1,), name='slot_id')
    layer_slot_id_emb = tf.keras.layers.Embedding(164, 32)
    slot_id_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_slot_id_emb(slot_id))

    channel_id = tf.keras.layers.Input(shape=(1,), name='channel_id')
    layer_channel_id_emb = tf.keras.layers.Embedding(342, 32)
    channel_id_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_channel_id_emb(channel_id))

    expose_cont = tf.keras.layers.Input(shape=(1,), name='expose_cont')

    title = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='title')
    title_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_words_emb(title))


    n_rate = tf.keras.layers.Input(shape=(1,), name='n_rate')
    v_rate = tf.keras.layers.Input(shape=(1,), name='v_rate')
    ne_rate = tf.keras.layers.Input(shape=(1,), name='ne_rate')
    add_date = tf.keras.layers.Input(shape=(1,), name='add_date')
    post_date = tf.keras.layers.Input(shape=(1,), name='post_date')
    click_cont = tf.keras.layers.Input(shape=(1,), name='click_cont')
    query_cont = tf.keras.layers.Input(shape=(1,), name='query_cont')


    input_concat = tf.keras.layers.Concatenate()([
        keywords_emb,
        disp_cate1_emb,
        disp_cate2_emb,
        disp_local1_emb,
        disp_local2_emb,
        slot_id_emb,
        channel_id_emb,
        expose_cont,
        title_emb,
        n_rate,
        v_rate,
        ne_rate,
        add_date,
        post_date,
        click_cont,
        query_cont
    ])

    probs = tf.keras.layers.Dense(2, activation='softmax',name='prbs')(input_concat)

    model = tf.keras.models.Model([
        keywords,
        disp_cate1,
        disp_cate2,
        disp_local1,
        disp_local2,
        slot_id,
        channel_id,
        expose_cont,
        title,
        n_rate,
        v_rate,
        ne_rate,
        add_date,
        post_date,
        click_cont,
        query_cont
    ], probs, name='isclk')

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

    # 这里ori_path相关代码是为了获取每一部分数据集的具体名称（和原始数据有关）
    ori_path = '/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_merge_all_rt_ljq/dt=' + flags_obj.date
    os.system('hadoop fs -mkdir /home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + flags_obj.date)
    name_list = tf.io.gfile.listdir('hdfs://default' + ori_path)

    train_dateset_itrators = []
    vali_dateset_itrators = []
    for index in range(0, 24):
    # for index in range(0, 1):
        for rg in range(0, 3000000, 200000):
        # for rg in range(0, 3, 200000):
            dest_name = ''
            for ind, nm in enumerate(name_list):
                if ind == index:
                    dest_name = nm
                    break
            appendix = flags_obj.date + '/' + str(index) + '_' + dest_name.split('.')[0] + '_' + str(rg)
            data_dir = flags_obj.data_dir + appendix
            if tf.io.gfile.exists(data_dir+'/my_hy_dataset_multi_process/0.1.0'):
                ds = tfds.builder('my_hy_dataset_multi_process', data_dir=data_dir)

                mnist_train, mnist_vali = ds.as_dataset(
                    split=['train', 'test'],
                )

                train_input_dataset = mnist_train.repeat().shuffle(buffer_size=100000).batch(128)
                eval_input_dataset = mnist_vali.repeat().batch(128)
                train_dateset_itrators.append(train_input_dataset.as_numpy_iterator())
                vali_dateset_itrators.append(eval_input_dataset.as_numpy_iterator())

    with strategy_scope:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0001, decay_steps=10000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model = build_model()

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics={'prbs':'sparse_categorical_accuracy'},
            # metrics={'prbs':tf.keras.metrics.AUC()}
        )
        checkpoint = tf.train.Checkpoint(myModel=model, myOptimizer=optimizer)
        manager = tf.train.CheckpointManager(checkpoint, directory='./hy_model', checkpoint_name='model.ckpt',max_to_keep=5)
        manager.restore_or_initialize()
    truths = []
    predicts = []
    for indx in range(0,100000000):
        i = indx % len(train_dateset_itrators)
        train_data = train_dateset_itrators[i].next()
        print('####################################### train %d start #############################################'%indx)
        print(model.train_on_batch(
            x=[
                train_data['keywords'], train_data['disp_cate1'], train_data['disp_cate2'], train_data['disp_local1'],
                train_data['disp_local2'],
                train_data['slot_id'], train_data['channel_id'], train_data['expose_cont'], train_data['title'],
                train_data['n_rate'],
                train_data['v_rate'], train_data['ne_rate'], train_data['add_date'], train_data['post_date'],
                train_data['click_cont'],
                train_data['query_cont']
            ],
            y=train_data['isclk'],
            # class_weight={0: 1, 1: 1}
        ))
        print('####################################### train %d end ###############################################'%indx)
        if indx % 100 == 0:
            vali_data = vali_dateset_itrators[i].next()
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ vali %d  start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'%indx)
            print(model.test_on_batch(
                x=[
                    vali_data['keywords'], vali_data['disp_cate1'], vali_data['disp_cate2'],
                    vali_data['disp_local1'],
                    vali_data['disp_local2'],
                    vali_data['slot_id'], vali_data['channel_id'], vali_data['expose_cont'], vali_data['title'],
                    vali_data['n_rate'],
                    vali_data['v_rate'], vali_data['ne_rate'], vali_data['add_date'], vali_data['post_date'],
                    vali_data['click_cont'],
                    vali_data['query_cont']
                ],
                y=vali_data['isclk'],
                # class_weight={0: 1, 1: 1}
            ))

            outputs = model.predict_on_batch(
                x=[
                    vali_data['keywords'], vali_data['disp_cate1'], vali_data['disp_cate2'],
                    vali_data['disp_local1'],
                    vali_data['disp_local2'],
                    vali_data['slot_id'], vali_data['channel_id'], vali_data['expose_cont'], vali_data['title'],
                    vali_data['n_rate'],
                    vali_data['v_rate'], vali_data['ne_rate'], vali_data['add_date'], vali_data['post_date'],
                    vali_data['click_cont'],
                    vali_data['query_cont']
                ],
            )
            truth = vali_data['isclk']
            predict = np.argmax(outputs, axis=-1)

            truths = truths+list(truth)
            predicts = predicts+list(predict)

            if len(truths)>1000000:
                truths = truths[-100000:]
                predicts = predicts[-100000:]

            fpr, tpr, thresholds = metrics.roc_curve(truths, predicts, pos_label=1)
            print('total auc %f'%metrics.auc(fpr, tpr))

            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ vali %d  end @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'%indx)
            manager.save()
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
    FLAGS.set_default('data_dir',  'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/')
    FLAGS.set_default('model_dir', 'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/model')
    FLAGS.set_default('distribution_strategy', r'one_device')
    FLAGS.set_default('num_gpus',0)
    FLAGS.set_default('train_epochs', 100)
    flags.DEFINE_bool('download', True,'Whether to download data to `--data_dir`.')
    flags.DEFINE_string('date', '20200913', 'which date `--date`.')


# %%
def main(_):
    stats = run(flags.FLAGS)
    logging.info('Run stats:\n%s', stats)


# %%

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_mnist_flags()
    app.run(main)

