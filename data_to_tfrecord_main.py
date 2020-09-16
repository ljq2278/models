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

import os
import sys
import tensorflow as tf
import subprocess
import time
import json

date = sys.argv[1]
indexs = sys.argv[2].split(',')

path = '/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/hy_merge_all_rt_ljq/dt=' + date

os.system('hadoop fs -mkdir /home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + date)
name_list = tf.io.gfile.listdir('hdfs://default' + path)

for index_str in indexs:
    index = int(index_str)
    dest_name = ''
    for ind, nm in enumerate(name_list):
        if ind == index:
            dest_name = nm
            break

    if not os.path.isfile(date + '_' + str(index) + '_' + dest_name):
        os.system('hadoop fs -text ' + path + '/' + dest_name + '>' + date + '_' + str(index) + '_' + dest_name)

    ps = []
    for rg in range(0, 3000000, 200000):

        if index >= len(name_list):
            print('index %d too big'%index)
            continue


        f = open('/code/tf_official_codes/my_hy_dataset_param.json', 'w', encoding='utf-8')
        dict_wr = {}
        dict_wr["date"] = date
        dict_wr["index"] = str(index)
        dict_wr["dest_name"] = dest_name
        dict_wr["train_range"] = str(rg) + '_' + str(rg + 160000)
        dict_wr["vali_range"] = str(rg + 160000) + '_' + str(rg + 200000)
        json.dump(dict_wr, f)
        f.close()

        dest_name = name_list[index]

        appendix = date + '/' + str(index) + '_' + dest_name.split('.')[0] + '_' + str(rg)
        data_dir = 'hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/demo/hy/' + appendix
        print(data_dir)
        ps.append(subprocess.Popen('python data_to_tfrecord_partprocess.py %s'%data_dir,shell=True))
        time.sleep(5)
        print('start %d'%rg)

    over = False

    while not over:
        time.sleep(5)
        none_cont = 0
        for p in ps:
            ret = p.poll()
            if ret is None:
                none_cont += 1
        print(none_cont)
        if none_cont==0:
            over = True
    print('over index %d'%index)
    os.system('rm '+ date + '_' + str(index) + '_' + dest_name)