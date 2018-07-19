import numpy as np
import tensorflow as tf

import vgg16
import utils

import cv2


img1 = cv2.imread("/home/rocket/Downloads/pic1.jpeg")
img2 = cv2.imread("/home/rocket/Downloads/pic2.jpeg")

s1 = img1.shape
s2 = img1.shape

batch1 = img1.reshape((1, s1[0], s1[1], s1[2]))
batch2 = img1[:,::-1,:].reshape((1, s2[0], s2[1], s2[2]))
# cv2.imshow('i2',img1[:,::-1,:]);
# cv2.waitKey(6000)
#
# cv2.imshow('i2',img1);
#
# cv2.waitKey(6000)

batch = np.concatenate((batch1,batch2 ), 0)
#batch2 = np.concatenate((batch1, ), 0)

def perceptual_loss(images):
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    features_vgg = vgg.conv4_3
    sub = features_vgg[0,:,:,:]-features_vgg[1,:,:,:]
    return sub

if __name__=="__main__":
    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        with tf.Session() as sess:
            images = tf.placeholder(np.float32, [None, None,None, None])
            feed_dict = {images: batch}
            sub = perceptual_loss(batch)
            sub_v = sess.run(sub, feed_dict=feed_dict)
            print(sub_v)
            print(sub_v.shape)



