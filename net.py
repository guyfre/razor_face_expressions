from slim.nets.resnet_v2 import resnet_v2_50
import tensorflow as tf
import face_data_util
import matplotlib.pyplot as plt
import tensorflow.contrib as con
import numpy as np


class FerNet(object):

    def __init__(self,num_classes=7):
        self.num_classes=num_classes
        self.image_size = 48
        self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.labels = tf.placeholder(tf.float32, [None,num_classes], name='labels')
        self.logits, self.end_points = resnet_v2_50(self.inputs, num_classes=num_classes)
        self.sess = tf.InteractiveSession()
        # get the last layer before the logits
        embedding_layer = self.end_points['global_pool']
        # normalize it
        embedding_layer = tf.squeeze(embedding_layer, axis=[1, 2])
        embedding_layer = tf.nn.l2_normalize(embedding_layer, axis=1, name='embed_normalized')
        # calc combined loss
        with tf.name_scope('loss'):
            dml_loss = con.losses.metric_learning.triplet_semihard_loss(self.labels, embedding_layer, margin=0.5)
            dml_loss = tf.Print(dml_loss, [dml_loss], "dml_loss: ", first_n=3, name='dml_loss')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels,
                                                                    name='cross_entropy_loss')
            cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross_entropy: ", first_n=3)
            alpha = 1
            beta = 1
            l2_loss = tf.losses.get_regularization_loss(name='l2_loss')
            total_loss = tf.add(alpha * dml_loss + beta * cross_entropy, l2_loss, name='total_loss')
            self.total_loss = tf.Print(total_loss, [total_loss], "total_loss: ", first_n=3)

        # accuracy of the trained model, between 0 (worst) and 1 (best)
        check_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(check_predictions, tf.float32))
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        # define the optimizer
        self.lr = tf.Variable(1e-4, trainable=False)
        self.train_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.total_loss,
                                                                                           global_step=self.gstep)
        # init the net
        self.sess.run(tf.global_variables_initializer())

        self.writer_train = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
        self.writer_test = tf.summary.FileWriter('./graphs/test', tf.get_default_graph())

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.total_loss)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()


    def train(self,batch_size=32,epochs=1):

        for epoch in range(epochs):
            train_generator = face_data_util.get_next_batch('/home/rocket/PycharmProjects/test/FERPlus/data',
                                                            '/home/rocket/PycharmProjects/test/FERPlus/data_base_dir',
                                                            mode='train', batch_size=batch_size)
            while True:
                try:
                    batch_x, batch_y = next(train_generator)
                except:
                    # stop iteration
                    break;




    def close(self):
        self.writer_train.close()
        self.writer_test.close()
        self.sess.close()



net = FerNet()


net.close()



