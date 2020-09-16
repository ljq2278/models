import numpy as np

import tensorflow as tf

import cv2

def get_result(img):
    # tf.gfile = tf.io.gfile
    mnist_model = tf.saved_model.load(r'model\mnist')

    print(mnist_model.signatures['serving_default'].inputs)

    print(mnist_model.signatures['serving_default'].output_dtypes)

    print(mnist_model.signatures['serving_default'].output_shapes)


    # 多输入的情况
    # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #
    # image3 = cv2.resize(img,(3,3))/255.0
    # image_np3 = image3[np.newaxis,...,np.newaxis]
    # input_tensor3 = tf.convert_to_tensor(image_np3,dtype=tf.float32)
    #
    # image6 = cv2.resize(img,(6,6))/255.0
    # image_np6 = image6[np.newaxis,...,np.newaxis]
    # input_tensor6 = tf.convert_to_tensor(image_np6,dtype=tf.float32)
    #
    # image12 = cv2.resize(img,(12,12))/255.0
    # image_np12 = image12[np.newaxis,...,np.newaxis]
    # input_tensor12 = tf.convert_to_tensor(image_np12,dtype=tf.float32)


    # Run inference
    # model_fn = mnist_model.signatures['serving_default']
    #
    # output_dict = model_fn(
    #     image12=input_tensor12,
    #     image3=input_tensor3,
    #     image6=input_tensor6,
    #     road=input_tensor3,
    #     roadExt=input_tensor3
    # )
    # print(np.argmax(output_dict['dense_1'],axis=1))

    # mnist预测

    image = 1-(cv2.resize(img,(28,28))/255.0)
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap = 'gray', interpolation='bicubic')

    image_np = image[...,np.newaxis]
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np,dtype=tf.float32)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = mnist_model.signatures['serving_default']

    output_dict = model_fn(input_1=input_tensor)
    # print(np.argmax(output_dict['dense_1'],axis=1))
    return np.argmax(output_dict['dense_1'],axis=1)