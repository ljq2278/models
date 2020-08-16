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


class MyDataset(tfds.core.GeneratorBasedBuilder):
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
            "image3": tfds.features.Tensor(shape=[3,3,1],dtype=tf.float64),
            "image6": tfds.features.Tensor(shape=[6, 6, 1],dtype=tf.float64),
            "image12": tfds.features.Tensor(shape=[12, 12, 1],dtype=tf.float64),
            "road": tfds.features.Tensor(shape=[3, 3, 1],dtype=tf.float64),
            "roadExt": tfds.features.Tensor(shape=[3, 3, 1],dtype=tf.float64),
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
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                fnames=['0_000001_3.json','1_000015_5.json','2_000006_4.json'],
                path=r'F:\dataset\AMAP-TECH\json_feature',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                fnames=['0_000002_4.json', '1_000024_3.json', '2_000007_4.json'],
                path=r'F:\dataset\AMAP-TECH\json_feature',
            )),
    ]

  def _generate_examples(self,fnames,path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    data = []
    for fname in fnames:
        info = json.load(open(os.path.join(path,fname),'r',encoding='utf-8'))
        data.append([
            np.array(info['car_3'])[:,:,np.newaxis],
            np.array(info['car_6'])[:,:,np.newaxis],
            np.array(info['car_12'])[:,:,np.newaxis],
            np.array(info['road_1'])[:,:,np.newaxis],
            np.array(info['road_2'])[:,:,np.newaxis],
            int(fname.split('_')[0])
        ])

    for index, (image3,image6,image12,road,roadExt,label) in enumerate(data):
        record = {"image3": image3,"image6": image6,"image12": image12,"road": road,"roadExt": roadExt, "label": label}
        yield index, record

