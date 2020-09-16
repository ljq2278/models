
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils

keywords = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,name='keywords')
print(keywords.shape)
layer_words_emb = tf.keras.layers.Embedding(3966, 32)
keywords_emb = tf.keras.layers.GlobalAveragePooling1D()(layer_words_emb(keywords))
print(layer_words_emb(keywords).shape)
print(keywords_emb.shape)