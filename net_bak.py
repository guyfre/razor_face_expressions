import tensorflow as tf
import face_data_util
import matplotlib.pyplot as plt
import tensorflow.contrib as con
from slim.nets import resnet_v2
import tensorflow.contrib.slim as slim

# important note: i use slim from 2 different sources: tf.contrib.slim and the one i have as source in models

import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image


tf.app.flags.DEFINE_string('inputs_path', '/home/rocket/PycharmProjects/test/FERPlus/data_base_dir',
                           'path to input images folder')
tf.app.flags.DEFINE_string('labels_path', '/home/rocket/PycharmProjects/test/FERPlus/data', 'path to labels folder')
tf.app.flags.DEFINE_string('outputs_path', './outputs', 'path to program outputs folder')
tf.app.flags.DEFINE_integer('num_classes', 7, 'number of classes to classify')
tf.app.flags.DEFINE_integer('batch_size', 128, 'train batch size')
tf.app.flags.DEFINE_integer('val_batch_size', 256, 'val batch size')
tf.app.flags.DEFINE_integer('image_size', 48, 'image size')
tf.app.flags.DEFINE_integer('max_epochs', 1000, 'num of epochs')
tf.app.flags.DEFINE_integer('save_epoch', 3, 'save checkpoint after n epochs')
tf.app.flags.DEFINE_float('lr_init', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.5, 'learning rate decay factor')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'num steps to decay lr')
tf.app.flags.DEFINE_float('save_images', 1, 'number of epochs to save images')

FLAGS = tf.app.flags.FLAGS


class FerNet(object):

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 1], name='input')
        self.inputs = self.preprocess(self.inputs)
        self.labels = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='labels')
        # scope = 'net'
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            self.logits, self.end_points = resnet_v2.resnet_v2_50(self.inputs, num_classes=FLAGS.num_classes)
        self.sess = tf.InteractiveSession()
        # get the last layer before the logits
        embedding_layer = self.end_points['global_pool']
        # normalize it
        embedding_layer = tf.squeeze(embedding_layer, axis=[1, 2])
        embedding_layer = tf.nn.l2_normalize(embedding_layer, axis=1, name='embed_normalized')
        # calc combined loss
        with tf.name_scope('loss'):
            max_labels = tf.argmax(self.labels, axis=1)
            dml_loss = con.losses.metric_learning.triplet_semihard_loss(max_labels, embedding_layer, margin=0.5)
            # dml_loss = tf.Print(dml_loss, [dml_loss], "dml_loss: ", first_n=3, summarize=50, name='dml_loss')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels,
                                                                       name='cross_entropy_loss')
            cross_entropy = tf.reduce_mean(cross_entropy)
            # cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross_entropy: ", first_n=3, summarize=50)
            alpha = 1.
            beta = 1.
            l2_loss = tf.losses.get_regularization_loss(name='l2_loss')
            self.total_loss = tf.add(alpha * dml_loss + beta * cross_entropy, l2_loss, name='total_loss')
            # self.total_loss = tf.Print(total_loss, [total_loss], "total_loss: ", first_n=3, summarize=50,
            #                            name='total_loss')

        pred_max = tf.argmax(self.end_points['predictions'], 1)
        label_max = tf.argmax(self.labels, 1)
        self.correct_predictions = tf.equal(pred_max, label_max)
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        # check_predictions = tf.equal(pred_max, label_max)
        # accuracy = tf.reduce_mean(tf.cast(check_predictions, tf.float32))
        # self.accuracy = tf.Print(accuracy, [accuracy, self.labels, self.end_points['predictions']], 'my acc/l/p: ',
        #                          summarize=50)
        metric_vars_scope_name = 'my_metrics'
        with tf.variable_scope(metric_vars_scope_name):
            self.acc, acc_op = tf.metrics.accuracy(label_max, pred_max)
            kappa, kappa_op = con.metrics.cohen_kappa(label_max, pred_max, FLAGS.num_classes)
            rec, rec_op = tf.metrics.recall(label_max, pred_max)
            pre, pre_op = tf.metrics.precision(label_max, pred_max)
            f1 = 2 * pre * rec / (pre + rec)

        self.all_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=metric_vars_scope_name)
        self.metrics_init_op = tf.variables_initializer(var_list=self.all_metric_vars,
                                                        name='validation_metrics_init')
        self.metric_ops = tf.group(acc_op, kappa_op, rec_op, pre_op)

        # confusing metrix
        self.cm_labels = tf.placeholder(tf.int8, name='cm_labels')
        self.cm_preds = tf.placeholder(tf.int8, name='cm_preds')
        self.cm = tf.confusion_matrix(self.cm_labels,
                                      self.cm_preds,
                                      num_classes=FLAGS.num_classes, name='confusion_matrix')

        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        # define the optimizer
        lr = tf.train.exponential_decay(FLAGS.lr_init,
                                        self.gstep,
                                        FLAGS.decay_steps,
                                        FLAGS.lr_decay,
                                        staircase=True,
                                        name='exponential_decay_learning_rate')

        self.train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.total_loss,
                                                                           global_step=self.gstep)

        self.pred_max = pred_max
        self.label_max = label_max
        # init the net
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.outputs_path, 'train'), tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(os.path.join(FLAGS.outputs_path, 'test'), tf.get_default_graph())

        tf.summary.scalar('Accuracy', self.acc)
        tf.summary.scalar('Total_loss', self.total_loss)
        tf.summary.scalar('dml_loss', dml_loss)
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('cross_entropy_loss', cross_entropy)
        tf.summary.scalar('Cohen-Kappa', kappa)
        tf.summary.scalar('Recall', rec)
        tf.summary.scalar('Precision', pre)
        tf.summary.scalar('F1', f1)

        self.merged = tf.summary.merge_all()

        # tensor board
        self.tb_images = tf.placeholder(tf.float32, [None, None, None, 1], name='tb_images')
        self.img_summary = tf.summary.image('wrong_images', self.tb_images, max_outputs=50)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

    def preprocess(self, batch):
        with tf.name_scope('preprocess'):
            new_batch = tf.divide(batch, 255)
            new_batch = new_batch - 0.5
            new_batch = new_batch * 2
        return new_batch

    def get_misclassified_images(self, batch_x, batch_y, limit_size=300):
        if batch_y.shape[0] > limit_size:
            # choose batch randomly
            idx = np.random.choice(range(batch_y.shape[0]), limit_size)
            batch_x = batch_x[idx]
            batch_y = batch_y[idx]
        preds, labels, predictions_bool = self.sess.run([self.pred_max, self.label_max, self.correct_predictions],
                                                        feed_dict={self.inputs: batch_x, self.labels: batch_y})
        wrong_predictions = [count for count, p in enumerate(predictions_bool) if not p]
        space_height, space_width = 20, 140
        images = np.zeros((len(wrong_predictions), FLAGS.image_size + space_height, FLAGS.image_size + space_width))
        for count, index in enumerate(wrong_predictions):
            im = np.squeeze(batch_x[index])
            images[count][space_height:, space_width / 2:-space_width / 2] = im
            pred = preds[index]
            label = labels[index]
            pil_im = Image.fromarray(images[count])
            draw = ImageDraw.Draw(pil_im)
            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
            pred_txt = 'P:' + face_data_util.labels_text[pred]
            lbl_txt = 'L:' + face_data_util.labels_text[label]
            draw.text((0, 0), pred_txt + '/' + lbl_txt, font=fnt)
            images[count] = np.asarray(pil_im)
        return images

    def batch_loss_acc(self, feed_dict):
        # reset the local variables, update all metrics and return the result of loss and acc

        # Reset the running variables
        self.sess.run(self.metrics_init_op)

        # Update the running variables on new batch of samples
        # save res for debugging
        res = self.sess.run(self.metric_ops, feed_dict=feed_dict)

        # Calculate the acc and loss on this batch
        acc, loss = self.sess.run([self.acc, self.total_loss], feed_dict=feed_dict)

        return loss, acc

    def get_confusion_metrix(self, mode='val'):
        generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                  FLAGS.inputs_path,
                                                  mode=mode, batch_size=FLAGS.val_batch_size, verbose=False)
        gt = []
        preds = []
        while True:
            try:
                x, y = next(generator)
                y = y / 10.
                gt.append(y.argmax(axis=1))
                if len(x.shape) == 3:
                    # we need to add axis
                    x = np.expand_dims(x, axis=3)
            except:
                break
        # flatten those array
        gt = np.array(gt).reshape(-1)
        preds = np.array(preds).reshape(-1)
        # get the matrix
        matrix = self.sess.run(self.cm, feed_dict={self.cm_labels: gt, self.cm_preds: preds})
        return matrix

    def eval_net(self, is_all=False):
        # validation set
        val_generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                      FLAGS.inputs_path,
                                                      mode='val', batch_size=FLAGS.val_batch_size, verbose=False)
        losses = []
        accs = []
        merges = []
        total_x = []
        total_y = []
        while True:
            try:
                val_x, val_y = next(val_generator)
                val_y = val_y / 10.
                if len(val_x.shape) == 3:
                    # we need to add axis
                    val_x = np.expand_dims(val_x, axis=3)
                feed_dict = {self.inputs: val_x, self.labels: val_y}
                loss, acc = self.batch_loss_acc(feed_dict)
                merged = self.sess.run(self.merged, feed_dict=feed_dict)
                losses.append(loss)
                accs.append(acc)
                merges.append(merged)
                total_x.append(val_x)
                total_y.append(val_y)
                if not is_all:
                    # finish only after one iteration
                    break
            except:
                break
        acc = np.mean(accs)
        loss = np.mean(losses)
        total_x = np.concatenate(total_x, axis=0)
        total_y = np.concatenate(total_y, axis=0)
        return total_x, total_y, acc, loss, merges[-1]

    def train(self):
        step = self.gstep.eval()
        verbose = True

        for epoch in range(FLAGS.max_epochs):

            train_generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                            FLAGS.inputs_path,
                                                            mode='train', batch_size=FLAGS.batch_size, verbose=verbose)
            if epoch == 0:
                verbose = False
            print('Epoch %d:' % epoch)
            while True:
                try:
                    batch_x, batch_y = next(train_generator)
                    batch_y = batch_y / 10.
                    if len(batch_x.shape) == 3:
                        # we need to add axis
                        batch_x = np.expand_dims(batch_x, axis=3)
                except:
                    # stop ite
                    break

                # run the train operation
                feed_dict = {self.inputs: batch_x, self.labels: batch_y}
                self.sess.run(self.train_opt, feed_dict=feed_dict)
                # reset and update the batch metrics and get loss and acc
                loss, acc = self.batch_loss_acc(feed_dict)
                # calc all the metric values for writing
                _merged = self.sess.run(self.merged, feed_dict=feed_dict)
                if step % 100 == 0:
                    print('step {} - loss: {}, acc: {}'.format(step, loss, acc))
                step += 1
                self.train_writer.add_summary(_merged, step)

            if (epoch % FLAGS.save_epoch) == 0:
                self.saver.save(self.sess, FLAGS.outputs_path + '/weights/epoch', epoch)
            # calc validation accuracy
            val_x, val_y, acc, loss, _merged = self.eval_net(is_all=True)
            self.test_writer.add_summary(_merged, step)
            # for debugging:
            loss_value, acc_value = self.batch_loss_acc(feed_dict={self.inputs: batch_x, self.labels: batch_y})
            print('-----Epoch %d summary----- \nTrain - loss: %.3f, acc: %.3f\nTest - loss: %.3f, acc: %.3f' % (
                epoch, loss_value, acc_value, loss, acc))
            if epoch % FLAGS.save_images == 0:
                # save wrong images to tensorboard
                wrong_images = self.get_misclassified_images(val_x, val_y)
                wrong_images = np.expand_dims(wrong_images, axis=3)
                image_summary = self.sess.run(self.img_summary, feed_dict={self.tb_images: wrong_images})
                self.train_writer.add_summary(image_summary, step)

    def close(self):
        self.train_writer.close()
        self.test_writer.close()
        self.sess.close()


# main
net = FerNet()
net.train()
net.close()
