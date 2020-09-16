################################################################################################

# 离线任务运行要做的准备工作


import os

# 相关包的安装

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

import tensorflow_datasets as tfds
from multiprocessing import Process
import threading
import time


def data_to_tfrecord(appendix):
    ds = tfds.builder('my_hy_dataset_multi_process',data_dir='hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + appendix)
    ds.download_and_prepare()


import sys

# dataQueue = Queue(30)  # max 50 images in queue
# dataPreparation = [None] *24
# if __name__ == '__main__':
for j in range(0,2):
    # f = os.popen('pwd')
    # print(f.readlines())
    # f.close()
    prs = []
    date = '20200913'
    index = int(sys.argv[1])+j
    path = '/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_merge_all_rt_ljq/dt='+date

    os.system('hadoop fs -mkdir /home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + date)
    name_list = tf.io.gfile.listdir('hdfs://default'+path)
    dest_name = ''
    for ind, nm in enumerate(name_list):
        if ind == index:
            dest_name = nm
            break
    if not os.path.isfile(date+'_'+str(index)+'_'+dest_name):
        os.system('hadoop fs -text '+path+'/'+dest_name+'>'+date+'_'+str(index)+'_'+dest_name)
    for rg in range(0, 3000000, 200000):
        f = open('/code/tf_official_codes/my_hy_dataset_param.json', 'w', encoding='utf-8')
        dict_wr = {}
        dict_wr["date"] = date
        dict_wr["index"] = str(index)
        dict_wr["dest_name"] = dest_name
        dict_wr["train_range"] = str(rg) + '_' + str(rg + 160000)
        dict_wr["vali_range"] = str(rg + 160000) + '_' + str(rg + 200000)
        import json
        json.dump(dict_wr,f)
        f.close()
        appendix = date + '/' + str(index) + '_' + dest_name.split('.')[0]+'_' +str(rg)
        # ds = tfds.builder('my_hy_dataset_multi_process',data_dir='hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + appendix)
        # ds.download_and_prepare()
        t = threading.Thread(target=data_to_tfrecord, args=(appendix,))
        t.start()
        # process = Process(target=data_to_tfrecord, args=(appendix,))
        # process.daemon = True
        # process.start()
        print('process start')
        prs.append(t)
        time.sleep(1)

    for t in prs:
        t.join()
    os.system('rm '+date + '_' + str(index) + '_' + dest_name)