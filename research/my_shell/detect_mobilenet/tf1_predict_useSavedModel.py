import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import skimage
from skimage import data,exposure
import matplotlib.pyplot as plt

class TOD(object):
    def __init__(self):
        # self.PATH_TO_CKPT = 'D:/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
        self.PATH_TO_CKPT2 = r'D:\models\ssd_mobilenet_v2_coco_2018_03_29\saved_model'
        self.PATH_TO_LABELS = r'D:\projects\tensorflowModelGarden\research\object_detection\data\mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 90

        self.category_index = self._load_label_map()


    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def get_detect_result(self, image):


        with tf.Session(graph=tf.Graph()) as sess:
            # tf.saved_model.load(sess, ["serve"], self.PATH_TO_CKPT2)
            tf.saved_model.load(sess, ["serving_default"], self.PATH_TO_CKPT2)
            graph = tf.get_default_graph()

            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes = graph.get_tensor_by_name('detection_boxes:0')
            scores =graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            # Actual detection.

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            image2 = vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            return image2



if __name__ == '__main__':
    image_ori = cv2.imread(r'D:\dataset\traffic_state_predict\amap_traffic_train_0712\000001\1_2019-03-10-18-08-08.jpg')

    # cv2.imshow("ori", image_ori)
    # cv2.waitKey(0)

    detecotr = TOD()
    image2 = detecotr.get_detect_result(image_ori)
    cv2.imshow("detection", image2)
    cv2.waitKey(0)



