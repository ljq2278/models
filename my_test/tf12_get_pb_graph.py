#python3
import os
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard

# #tf1
# model_path = r'D:/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
# import_to_tensorboard(model_dir=model_path, log_dir='log/')
#tf2
model_path = r'D:\models\mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8\saved_model'
import_to_tensorboard(model_dir=model_path, log_dir='log/',tag_set='serve')

# import tensorflow as tf
# from tensorflow.core.protobuf import saved_model_pb2
# from tensorflow.python.util import compat
# from tensorflow.python.framework import ops
#
# model_path = r"D:\models\mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8\saved_model\saved_model.pb"
# with tf.Session(graph=ops.Graph()) as sess:
#     with tf.gfile.GFile(model_path, "rb") as f:
#         data = compat.as_bytes(f.read())
#         sm = saved_model_pb2.SavedModel()
#         sm.ParseFromString(data)
#         g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
#         train_writer = tf.summary.FileWriter("./log")
#         train_writer.add_graph(sess.graph)
#         train_writer.flush()
#         train_writer.close()