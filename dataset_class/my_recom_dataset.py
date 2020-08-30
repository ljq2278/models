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


class MyRecomDataset(tfds.core.GeneratorBasedBuilder):
  """TODO(my_dataset): Short description of my dataset."""

  # TODO(my_dataset): Set up version.
  VERSION = tfds.core.Version('0.1.0')


  def _info(self):
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "input": tfds.features.Tensor(shape=[3,3,1],dtype=tf.float64),
            "label": tfds.features.ClassLabel(num_classes=3),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.

        # supervised_keys=("image3","image6","image12","road","roadExt", "label"),

        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    self.feat_name_candinum = [
        ['feat0', 3],
        ['feat1', 5],
        ['feat2', 10],
        ['feat3', 5],
    ]

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                path=r'd:\dataset\reconmand_sample\train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                path=r'd:\dataset\reconmand_sample\vali',
            )),
    ]

  def discre_to_onehot(self,discre,candinum):
      # 离散值从-1开始往上走，-1表示oov
      # ind从0开始往上走
      # candinum表示包含了oov的特征取值数
      ind = discre + 1
      ret = np.zeros(candinum)
      ret[ind] = 1
      return ret

  def _generate_examples(self,path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    fnames = os.listdir(path)
    data = []
    for fname in fnames:
        f = open(os.path.join(path,fname),'r',encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            info = json.loads(line.strip())
            data.append([
                np.concatenate(
                    [np.array(self.discre_to_onehot(info[self.feat_name_candinum[i][0]],self.feat_name_candinum[i][1]) for i in range(len(self.feat_name_candinum)))]
                ),
                int(info['label']),
            ])

    for index, (input,label) in enumerate(data):
        record = {"input": input,"label": label}
        yield index, record

