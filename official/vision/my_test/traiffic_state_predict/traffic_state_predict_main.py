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
from official.utils.misc import model_helpers
from official.vision.image_classification.resnet import common
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import json

FLAGS = flags.FLAGS


def build_model():
    """Constructs the ML model used to predict handwritten digits."""

    image12 = tf.keras.layers.Input(shape=(12, 12, 1), name='image12')
    image6 = tf.keras.layers.Input(shape=(6, 6, 1), name='image6')
    image3 = tf.keras.layers.Input(shape=(3, 3, 1), name='image3')
    road = tf.keras.layers.Input(shape=(3, 3, 1), name='road')
    roadExt = tf.keras.layers.Input(shape=(3, 3, 1), name='roadExt')

    y = tf.keras.layers.Conv2D(filters=8,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(image12)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='same')(y)
    y = tf.keras.layers.concatenate([y, image6], axis=-1)

    y = tf.keras.layers.Conv2D(filters=12,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='same')(y)
    y = tf.keras.layers.concatenate([y, image3, road, roadExt], axis=-1)

    y = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(3, 3),
                                     padding='same')(y)

    y = tf.keras.layers.Flatten()(y)
    # y = tf.keras.layers.Dense(1024, activation='relu')(y)
    # y = tf.keras.layers.Dropout(0.4)(y)

    probs = tf.keras.layers.Dense(3, activation='softmax')(y)

    model = tf.keras.models.Model([image12, image6, image3, road, roadExt], probs, name='my_dataset')

    return model


def run(flags_obj, datasets_override=None, strategy_override=None):
    """Run MNIST model training and eval loop using native Keras APIs.

    Args:
      flags_obj: An object containing parsed flag values.
      datasets_override: A pair of `tf.data.Dataset` objects to train the model,
                         representing the train and test sets.
      strategy_override: A `tf.distribute.Strategy` object to use for model.

    Returns:
      Dictionary of training and eval stats.
    """
    strategy = strategy_override or distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        tpu_address=flags_obj.tpu)

    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    mnist = tfds.builder('traffic_state_predict', data_dir=flags_obj.data_dir)
    if flags_obj.download:
        mnist.download_and_prepare()

    # inputs = ["image12"]
    # outputs = ["label"]
    # mnist = mnist.map(lambda ex: ({i: ex[i] for i in inputs}, {o: ex[o] for o in outputs}))
    mnist_train, mnist_vali, mnist_test = datasets_override or mnist.as_dataset(
        split=['train', 'test','predict'],
        # decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
        # as_supervised=True
    )

    # train_input_dataset = mnist_train.cache().repeat(10000).shuffle(buffer_size=100000).batch(128)
    train_input_dataset = mnist_train.cache().batch(300)
    eval_input_dataset = mnist_vali.cache().batch(300)
    test_input_dataset = mnist_test.cache().batch(600)

    with strategy_scope:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.005, decay_steps=100000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model = build_model()

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
            # metrics=weighted_fi_score,
        )
        checkpoint = tf.train.Checkpoint(myModel=model,myOptimizer=optimizer)
        manager = tf.train.CheckpointManager(checkpoint, directory='./save0', checkpoint_name='model.ckpt', max_to_keep=5)
        manager.restore_or_initialize()

    best_score = 0.0
    res = {}
    for indx, train_data in enumerate(train_input_dataset.as_numpy_iterator()):
        print(model.train_on_batch(
            x=[train_data['image12'], train_data['image6'], train_data['image3'], train_data['road'],train_data['roadExt']],
            y=train_data['label'],
            class_weight={0: 1, 1: 3, 2: 6}
            # class_weight={0: 2, 1: 8}
        ))
        if indx % 200 == 0:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            res_vali = {}
            truth = []
            predict = []
            for indx, vali_data in enumerate(eval_input_dataset.as_numpy_iterator()):
                # print(model.test_on_batch(x=[vali_data['image12'], vali_data['image6'], vali_data['image3'], vali_data['road'],vali_data['roadExt']], y=vali_data['label']))
                outputs = model.predict_on_batch(x=[vali_data['image12'], vali_data['image6'], vali_data['image3'], vali_data['road'],vali_data['roadExt']])
                truth = truth+list(vali_data['label'])
                predict = predict+list(np.argmax(outputs, axis=-1))
                for fname,output in zip(vali_data['fname'],list(outputs)):
                    res_vali[fname.decode()] = ','.join([str(itm) for itm in output])
            json.dump(res_vali, open('res_vali.json', 'w', encoding='utf-8'))
            p_class, r_class, f_class, support_micro = precision_recall_fscore_support(truth,predict,labels=[0, 1,2])
            print('vali scores:')
            print(f_class)
            print(0.2 * f_class[0] + 0.2 * f_class[1] + 0.6 * f_class[2],best_score)
            ##################################################################################
            truth_train = []
            predict_train = []
            res_train = {}
            for indx, train_data in enumerate(train_input_dataset.as_numpy_iterator()):
                outputs_train = model.predict_on_batch(
                    x=[train_data['image12'], train_data['image6'], train_data['image3'], train_data['road'],
                       train_data['roadExt']])
                truth_train = truth_train + list(train_data['label'])
                predict_train = predict_train + list(np.argmax(outputs_train, axis=-1))
                for fname,output in zip(train_data['fname'],list(outputs_train)):
                    res_train[fname.decode()] = ','.join([str(itm) for itm in output])
                # if indx>=100:
                #     break
            json.dump(res_train, open('res_train.json', 'w', encoding='utf-8'))
            p_class_train, r_class_train, f_class_train, support_micro_train = precision_recall_fscore_support(truth_train, predict_train, labels=[0, 1, 2])
            print('train scores:')
            print(f_class_train)
            print(0.2 * f_class_train[0] + 0.2 * f_class_train[1] + 0.6 * f_class_train[2], best_score)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            if 0.2 * f_class[0] + 0.2 * f_class[1] + 0.6 * f_class[2] > best_score:
                manager.save()
                res_test = {}
                export_path = os.path.join(flags_obj.model_dir, 'saved_model')
                model.save(export_path, include_optimizer=False)
                for indx, test_data in enumerate(test_input_dataset.as_numpy_iterator()):
                    outputs_test = model.predict_on_batch(x=[test_data['image12'], test_data['image6'], test_data['image3'], test_data['road'],test_data['roadExt']])
                    for fname, output in zip(test_data['fname'], list(outputs_test)):
                        res_test[fname.decode()] = ','.join([str(itm) for itm in output])
                    # for ind, fname in enumerate(test_data['fname']):
                    #     res[fname.decode()] = str(np.argmax(outputs_test, axis=-1)[ind])
                json.dump(res_test,open('res_test.json','w',encoding='utf-8'))
                # json.dump(res,open('res.json','w',encoding='utf-8'))
                best_score = 0.2 * f_class[0] + 0.2 * f_class[1] + 0.6 * f_class[2]

    return 'over'


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
    flags.DEFINE_bool('download', False,
                      'Whether to download data to `--data_dir`.')
    # FLAGS.set_default('batch_size', 16)


def main(_):
    model_helpers.apply_clean(FLAGS)
    stats = run(flags.FLAGS)
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_mnist_flags()
    app.run(main)
