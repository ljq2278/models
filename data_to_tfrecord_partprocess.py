
import tensorflow_datasets as tfds

import sys


print('sub process start')
data_dir = sys.argv[1]


ds = tfds.builder('my_hy_dataset_multi_process',data_dir=data_dir)
ds.download_and_prepare()
print('sub process over')


