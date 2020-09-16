########################################################################
"""my_dataset dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import json
import os
import numpy as np
import tensorflow as tf

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""

# TODO(my_dataset):
_DESCRIPTION = """
"""


class MyHyDatasetMultiProcess(tfds.core.GeneratorBasedBuilder):
    """TODO(my_dataset): Short description of my dataset."""

    # TODO(my_dataset): Set up version.
    VERSION = tfds.core.Version('0.1.0')


    def _info(self):
        # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
        self.max_sentence_length = 64
        self.word_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/word_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.word_index_dict[word] = int(ind)

        self.disp_cate1_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/disp_cate1_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.disp_cate1_index_dict[word] = int(ind)

        self.disp_cate2_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/disp_cate2_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.disp_cate2_index_dict[word] = int(ind)

        self.disp_local1_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/disp_local1_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.disp_local1_index_dict[word] = int(ind)

        self.disp_local2_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/disp_local2_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.disp_local2_index_dict[word] = int(ind)

        self.slot_id_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/slot_id_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.slot_id_index_dict[word] = int(ind)

        self.channel_id_index_dict = {}
        f = tf.io.gfile.GFile(
            'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_dict/channel_id_index', 'r')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            word, ind = line.split('\t')
            self.channel_id_index_dict[word] = int(ind)

        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                "keywords": tfds.features.Tensor(shape=[None], dtype=tf.int32),
                "disp_cate1": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "disp_cate2": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "disp_local1": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "disp_local2": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "slot_id": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "channel_id": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                "expose_cont": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "title": tfds.features.Tensor(shape=[None], dtype=tf.int32),
                "n_rate": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "v_rate": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "ne_rate": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "add_date": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "post_date": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "click_cont": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "query_cont": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                "isclk": tfds.features.ClassLabel(num_classes=2),
            }),

        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(my_dataset): Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs
        f = open('/code/tf_official_codes/my_hy_dataset_param.json','r',encoding='utf-8')
        import json
        params = json.load(f)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    date=params['date'],
                    index=params['index'],
                    dest_name=params['dest_name'],
                    rage=params['train_range']
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    date=params['date'],
                    index=params['index'],
                    dest_name=params['dest_name'],
                    rage=params['vali_range']
                )),
        ]

    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        return False

    def _generate_examples(self, date,index,dest_name,rage):
        """Yields examples."""
        # TODO(my_dataset): Yields (key, example) tuples from the dataset
        cont = 0
        f = open(date+'_'+index+'_'+dest_name, 'r',encoding='utf-8')
        print('before read')
        lines = f.readlines()
        print('after read')
        f.close()
        rg_start, rg_end = rage.split('_')
        for line in lines[int(rg_start):int(rg_end)]:
            if len(line.split('\001')) != 20:
                continue
            sid, entity_id, keywords, isclk, disp_cate1, disp_cate2, disp_local1, disp_local2, slot_id, channel_id, expose_cont, title, n_rate, v_rate, ne_rate, add_date, post_date, click_cont, query_cont, ts_mint = line[:-1].split(
                '\001')
            # 1:10的负采样
            if int(isclk) == 0:
                if np.random.random() > 0.07:
                    continue
            resi = {
                # 'keywords': [self.word_index_dict[word] for word in keywords if
                #              word in self.word_index_dict.keys()],
                'keywords': [max(self.word_index_dict.values())+1 if ind >= len(keywords) else 0 if keywords[ind] not in self.word_index_dict.keys() else self.word_index_dict[keywords[ind]] for ind in range(0,self.max_sentence_length)],
                "disp_cate1": [0] if disp_cate1 not in self.disp_cate1_index_dict.keys() else [
                    self.disp_cate1_index_dict[disp_cate1]],
                "disp_cate2": [0] if disp_cate2 not in self.disp_cate2_index_dict.keys() else [
                    self.disp_cate2_index_dict[disp_cate2]],
                "disp_local1": [0] if disp_local1 not in self.disp_local1_index_dict.keys() else [
                    self.disp_local1_index_dict[disp_local1]],
                "disp_local2": [0] if disp_local2 not in self.disp_local2_index_dict.keys() else [
                    self.disp_local2_index_dict[disp_local2]],
                "slot_id": [0] if slot_id not in self.slot_id_index_dict.keys() else [
                    self.slot_id_index_dict[slot_id]],
                "channel_id": [0] if channel_id not in self.channel_id_index_dict.keys() else [
                    self.channel_id_index_dict[channel_id]],
                "expose_cont": [-1] if not self.is_number(expose_cont) else [float(expose_cont)],
                # "title": [self.word_index_dict[word] for word in title if word in self.word_index_dict.keys()],
                "title": [max(self.word_index_dict.values())+1 if ind >= len(title) else 0 if title[ind] not in self.word_index_dict.keys() else self.word_index_dict[title[ind]] for ind in range(0,self.max_sentence_length)],
                "n_rate": [-1] if not self.is_number(n_rate) else [float(n_rate)],
                "v_rate": [-1] if not self.is_number(v_rate) else [float(v_rate)],
                "ne_rate": [-1] if not self.is_number(ne_rate) else [float(ne_rate)],
                "add_date": [-1] if not self.is_number(add_date) else [float(add_date)],
                "post_date": [-1] if not self.is_number(post_date) else [float(post_date)],
                "click_cont": [-1] if not self.is_number(click_cont) else [float(click_cont)],
                "query_cont": [-1] if not self.is_number(query_cont) else [float(query_cont)],
                "isclk": int(isclk)
            }
            yield cont, resi
            cont += 1
#         for index, datai in enumerate(data):
#             record = datai
# #             print(datai)
#             yield index, record
