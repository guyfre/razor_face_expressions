import tensorflow as tf
from pygments.formatters import img

import face_data_util
import matplotlib.pyplot as plt
import tensorflow.contrib as con

from nets.nasnet.nasnet_utils import factorized_reduction
from slim.nets import resnet_v2
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tensorboard.plugins import projector

# important note: i use slim from 2 different sources: tf.contrib.slim and the one i have as source in models
import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.app.flags.DEFINE_string('inputs_path', '/home/rocket/PycharmProjects/test/FERPlus/data_base_dir',
                           'path to input images folder')
tf.app.flags.DEFINE_string('labels_path', '/home/rocket/PycharmProjects/test/FERPlus/data', 'path to labels folder')
tf.app.flags.DEFINE_string('outputs_path', './outputs', 'path to program outputs folder')
tf.app.flags.DEFINE_integer('num_classes', 7, 'number of classes to classify')
tf.app.flags.DEFINE_integer('batch_size', 400, 'train batch size')
tf.app.flags.DEFINE_integer('val_batch_size', 512, 'val batch size')
tf.app.flags.DEFINE_integer('emb_batch_size', 512, 'embedding batch size')
tf.app.flags.DEFINE_integer('image_size', 48, 'image size')
tf.app.flags.DEFINE_integer('max_epochs', 1000, 'num of epochs')
tf.app.flags.DEFINE_integer('save_epoch', 3, 'save checkpoint after n epochs')
tf.app.flags.DEFINE_float('lr_init', 1e-3, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'learning rate decay factor')
tf.app.flags.DEFINE_integer('decay_steps', 3000, 'num steps to decay lr')
tf.app.flags.DEFINE_float('save_images', 3, 'number of epochs to save images')

FLAGS = tf.app.flags.FLAGS


class FerNet(object):

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 1], name='input')
        self.labels = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='labels')
        # scope = 'net'
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            self.logits, self.end_points = resnet_v2.resnet_v2_50(self.inputs, num_classes=FLAGS.num_classes)
        self.sess = tf.InteractiveSession()

        # get the last layer before the logits
        embedding_layer = self.end_points['global_pool']
        # self.predictions = self.end_points['predictions']

        # # normalize it
        # self.embedding_layer = tf.squeeze(embedding_layer, axis=[1, 2])
        # embedding_layer = self.embedding_layer
        # embedding_layer = tf.nn.l2_normalize(embedding_layer, axis=1, name='embed_normalized')
        # self.emb_size = self.embedding_layer.shape.dims[-1].value

        max_labels = tf.argmax(self.labels, axis=1)

        with tf.name_scope('embeddings'):
            # only for dml
            embedding_layer = tf.squeeze(embedding_layer, axis=[1, 2])
            regulizer = tf.contrib.layers.l2_regularizer(0.0001)
            embedding_layer = tf.layers.dense(embedding_layer, 128, kernel_regularizer=regulizer)
            self.emb_size = embedding_layer.shape.dims[-1].value
            norm_emb = tf.nn.l2_normalize(embedding_layer, axis=1, name='embed_normalized')
            # add activation
            embedding_layer = tf.nn.relu6(embedding_layer)
            self.embedding_layer = embedding_layer

        # from here it is the classifier addition
        dml_loss = con.losses.metric_learning.triplet_semihard_loss(max_labels, norm_emb, margin=0.2)
        # drop = tf.layers.dropout(embedding_layer, rate=0.5, name='drop_before_logits')
        self.logits = tf.layers.dense(embedding_layer, FLAGS.num_classes,
                                      kernel_regularizer=regulizer, name='logits')
        class_weights = tf.constant([0.3, 0.3, 0.13, 0.13, 0.1,0.01,0.03])
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(self.labels,self.logits,class_weights)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
        cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        self.predictions = tf.nn.softmax(self.logits, name='predictions')
        l2_loss = tf.losses.get_regularization_loss(name='l2_loss')
        self.total_loss = cross_entropy + dml_loss + 0.01 * l2_loss



        pred_max = tf.argmax(self.predictions, 1)
        label_max = tf.argmax(self.labels, 1)
        self.correct_predictions = tf.equal(pred_max, label_max)

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
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

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

        # embeddings
        self.EMB_DIR = FLAGS.outputs_path + '/emb'
        self.emb_file_name = 'metadata.tsv'
        self.images_emb = tf.Variable(np.zeros((FLAGS.emb_batch_size, self.emb_size)), name='images_emb')
        self.sess.run(self.images_emb.initializer)
        self.emb_saver = tf.train.Saver([self.images_emb])
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = self.images_emb.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = self.emb_file_name
        # Saves a config file that TensorBoard will read during startup.
        emb_writer = tf.summary.FileWriter(self.EMB_DIR)
        projector.visualize_embeddings(emb_writer, config)

        if os.path.exists('./outputs/weights'):
            restorer = tf.train.Saver()
            restorer.restore(self.sess, tf.train.latest_checkpoint('./outputs/weights'))
            print('restored')

        self.saver = tf.train.Saver(max_to_keep=4)
        self.initialize_uninitialized_vars()

    def initialize_uninitialized_vars(self):
        from itertools import compress
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
                                            for var in global_vars])
        not_initialized_vars = list(compress(global_vars, is_not_initialized))

        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def preprocess(self, batch, drop=None):
        # remove the channel dim (1 because it is grey scale)
        if len(batch.shape) == 4:
            batch = np.squeeze(batch,axis=3)
        # Histogram equalization
        if batch.dtype != np.uint8:
            batch = (batch * 255).astype(np.uint8)
        new_batch = np.array([cv2.equalizeHist(img) for img in batch])
        # make values range [-1,1]
        new_batch = (new_batch / 255. - 0.5) * 2
        # drop
        if drop:
            probs = np.random.uniform(size=batch.shape)
            mask = (probs > drop).astype(np.int8)
            new_batch = new_batch * mask

        # flip random horizontally
        # batch_size = new_batch.shape[0]
        # idx = np.random.choice(range(batch_size), batch_size / 2)
        # new_batch[idx, :, :] = new_batch[idx, :, ::-1]

        # return the channel
        new_batch = np.expand_dims(new_batch, axis=3)
        return new_batch

    def get_all_features(self):
        generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                  FLAGS.inputs_path,
                                                  mode='train', batch_size=FLAGS.val_batch_size, verbose=True)
        features = []
        ys = np.array([])
        while True:
            try:
                x, y = next(generator)

                y = y / 10.
                max_y = np.argmax(y, axis=1)
                if max_y.size < FLAGS.val_batch_size:
                    continue
                ys = np.append(ys, max_y)
                x = self.preprocess(x)
                features.append(self.sess.run(self.embedding_layer, feed_dict={self.inputs: x}))
            except:
                break
        features = np.array(features).reshape(-1, 128)
        return features, ys

    def vis_emb(self, batch_x, batch_y, step):

        metadata = self.EMB_DIR + '/' + self.emb_file_name

        features = self.sess.run(self.embedding_layer, feed_dict={self.inputs: batch_x})
        size = batch_y.size

        if not os.path.exists(self.EMB_DIR):
            os.makedirs(self.EMB_DIR)

        with open(metadata, 'w') as metadata_file:
            for i in range(size):
                c = batch_y[i]
                metadata_file.write('{}\n'.format(c))

        ass_op = self.images_emb.assign(features)
        self.sess.run(ass_op)
        self.emb_saver.save(self.sess, self.EMB_DIR + '/images_emb.ckpt', global_step=step)

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

    def predict(self, batch_x):
        pred_max, preds = self.sess.run([self.pred_max, self.predictions], feed_dict={self.inputs: batch_x})
        pred_img_names = [face_data_util.labels_text[p] + '.jpeg' for p in pred_max]
        path_pre = './emojies/'
        pred_dict = {}
        for i in range(len(pred_img_names)):
            pred_dict[i] = {'emoji_path': path_pre + pred_img_names[i], 'preds': preds[i]}
        return pred_dict

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

    def get_confusion_metrix(self, mode='val', preds=None, labels=None):
        if preds is None or labels is None:
            generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                      FLAGS.inputs_path,
                                                      mode=mode, batch_size=FLAGS.val_batch_size, verbose=False)
            labels = []
            preds = []
            while True:
                try:
                    x, y = next(generator)
                    y = y / 10.
                    labels.append(y.argmax(axis=1))
                    if len(x.shape) == 3:
                        # we need to add axis
                        x = np.expand_dims(x, axis=3)
                except:
                    break
            # flatten those array
            labels = np.array(labels).reshape(-1)
            preds = np.array(preds).reshape(-1)
        # get the matrix
        matrix = self.sess.run(self.cm, feed_dict={self.cm_labels: labels, self.cm_preds: preds})
        # sums = np.sum(matrix,axis=1).astype(np.float32)
        # matrix=matrix/sums
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
                # if len(val_x.shape) == 3:
                #     # we need to add axis
                #     val_x = np.expand_dims(val_x, axis=3)
                val_x = self.preprocess(val_x)
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

    def match_emb_to_centroid(self, centroids_list, emb):
        # calc distance between emb to all the centroids and return the probs
        logits = np.mean(np.square(centroids_list - emb), axis=1)
        sum = np.sum(logits)
        probs = logits / sum
        return probs

    def dml_preds_labels(self, from_mode='val', is_all=False, is_probs=False):
        generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                  FLAGS.inputs_path,
                                                  mode=from_mode, batch_size=FLAGS.val_batch_size, verbose=False)
        centroids = self.get_centroids(from_mode=from_mode)
        preds = []
        labels = []
        while True:
            try:
                batch_x, batch_y = next(generator)
                batch_y = batch_y / 10.
                # if len(batch_x.shape) == 3:
                #     # we need to add axis
                #     batch_x = np.expand_dims(batch_x, axis=3)
                batch_x = self.preprocess(batch_x)
                feed_dict = {self.inputs: batch_x, self.labels: batch_y}
                emb_out = self.sess.run(self.embedding_layer, feed_dict=feed_dict)
                max_y = np.argmax(batch_y, axis=1)
                for i in range(max_y.size):
                    labels.append(max_y[i])
                    pred = self.match_emb_to_centroid(centroids, emb_out[i])
                    if not is_probs:
                        pred = np.argmin(pred)
                    preds.append(pred)

                if not is_all:
                    # finish only after one iteration
                    break
            except:
                break
        return np.array(preds), np.array(labels)

    def calc_dml_acc(self, from_mode='val', is_all=False):
        preds, labels = self.dml_preds_labels(from_mode=from_mode, is_all=is_all)
        matches = preds == labels
        acc = np.mean(matches)
        return acc

    def get_centroids(self, from_mode='val'):
        generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                  FLAGS.inputs_path,
                                                  mode=from_mode, batch_size=FLAGS.val_batch_size, verbose=False)
        cls_dict = {}
        for i in range(FLAGS.num_classes):
            cls_dict[i] = {'counter': 0, 'emb_sum': np.zeros(self.emb_size)}
        while True:
            try:
                batch_x, batch_y = next(generator)
                batch_y = batch_y / 10.
                # if len(batch_x.shape) == 3:
                #     # we need to add axis
                #     batch_x = np.expand_dims(batch_x, axis=3)
                batch_x = self.preprocess(batch_x)
                feed_dict = {self.inputs: batch_x, self.labels: batch_y}
                emb_out = self.sess.run(self.embedding_layer, feed_dict=feed_dict)
                max_y = np.argmax(batch_y, axis=1)
                for i in range(len(max_y)):
                    ind = max_y[i]
                    cls_dict[ind]['emb_sum'] = cls_dict[ind]['emb_sum'] + emb_out[i]
                    cls_dict[ind]['counter'] = cls_dict[ind]['counter'] + 1

            except:
                break
        # compute the averages (centroids)
        cls_centroids = []
        for i in range(FLAGS.num_classes):
            cls_centroids.append(cls_dict[i]['emb_sum'] / cls_dict[i]['counter'])

        return np.array(cls_centroids)

    def train(self):
        step = self.gstep.eval()
        verbose = True

        dml_acc_hist = []

        # for dml only
        emb_generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                      FLAGS.inputs_path,
                                                      mode='test', batch_size=FLAGS.emb_batch_size, verbose=False)
        emb_x, emb_y = next(emb_generator)
        emb_y = emb_y / 10.
        emb_x = self.preprocess(emb_x)
        # -----------

        for epoch in range(FLAGS.max_epochs):

            train_generator = face_data_util.get_next_batch(FLAGS.labels_path,
                                                            FLAGS.inputs_path,
                                                            mode='train', batch_size=FLAGS.batch_size, verbose=verbose)

            save_emb_flag = True
            if epoch == 0:
                verbose = False
            print('Epoch %d:' % epoch)
            while True:
                try:
                    batch_x, batch_y = next(train_generator)
                    batch_y = batch_y / 10.
                    # if len(batch_x.shape) == 3:
                    #     # we need to add axis
                    #     batch_x = np.expand_dims(batch_x, axis=3)
                    batch_x = self.preprocess(batch_x, drop=None)
                except:
                    # stop iter
                    break
                # save visualize embeddings
                if save_emb_flag and batch_y.shape[0] == FLAGS.batch_size:
                    self.vis_emb(emb_x, np.argmax(emb_y, axis=1), step)
                    save_emb_flag = False

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
                save_emb_flag = True

            # calc validation accuracy
            val_x, val_y, acc, loss, _merged = self.eval_net(is_all=True)
            self.test_writer.add_summary(_merged, step)
            # for debugging:
            loss_value, acc_value = self.batch_loss_acc(feed_dict={self.inputs: batch_x, self.labels: batch_y})
            print('-----Epoch %d summary----- \nTrain - loss: %.3f, acc: %.3f\nTest - loss: %.3f, acc: %.3f' % (
                epoch, loss_value, acc_value, loss, acc))

            # calc dml acc
            # dml_acc_hist.append(self.calc_dml_acc(from_mode='val', is_all=False))
            # print('DML acc: %.3f' % dml_acc_hist[-1])
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

    def eval_pics(self,image_paths=None,images=None):
        batch_x = []
        if images is None:
            for path in image_paths:
                try:
                    face = face_data_util.get_face(img_path=path)
                    batch_x.append(face)

                except Exception as e:
                    print(e)
        else:
            for img in images:
                try:
                    face = face_data_util.get_face(img=img)
                    batch_x.append(face)

                except Exception as e:
                    print(e)
        # after collecting all the images
        cols = len(batch_x)
        rows = 2
        if cols==0:
            print('no faces in all images')
            return
        batch_x = np.array(batch_x)
        pr_x = self.preprocess(batch_x)
        pred_dict = self.predict(pr_x)

        for i in range(cols):
            in_img = batch_x[i]
            emoji_img = plt.imread(pred_dict[i]['emoji_path'])/255.
            # round with format like .2f%
            p_probs = np.around(pred_dict[i]['preds'], decimals=2)
            plt.subplot(rows,cols,i+1)
            plt.imshow(in_img,cmap='gray')
            plt.subplot(rows, cols, cols+i + 1)
            plt.imshow(emoji_img)
            print('Img %d:\npreds: %s'%(i,p_probs))


    def print_preds(self,img_names,mode='test'):
        batch_x, batch_y = face_data_util.get_specific_batch(FLAGS.labels_path, FLAGS.inputs_path,
                                                             img_names,mode)
        pr_x = self.preprocess(batch_x)
        pred_dict = self.predict(pr_x)
        rows = 2
        cols = len(img_names)
        num_correct = 0
        for i in range(cols):
            in_img = batch_x[i]
            emoji_img = plt.imread(pred_dict[i]['emoji_path'])/255.
            l_probs = batch_y[i]/10.
            max_l = np.argmax(l_probs)
            # round with format like .2f%
            p_probs = np.around(pred_dict[i]['preds'], decimals=2)
            if max_l==np.argmax(p_probs):
                num_correct += 1
            label_txt = face_data_util.labels_text[max_l]
            plt.subplot(rows,cols,i+1)
            plt.title(label_txt)
            plt.imshow(in_img,cmap='gray')
            plt.subplot(rows, cols, cols+i + 1)
            plt.imshow(emoji_img)
            print('Img %d:\nlabels: %s , preds: %s'%(i,l_probs,p_probs))

        acc = num_correct/float(cols)
        print('Accuracy: %.2f'%acc)

# main
net = FerNet()

# net.train()

# for live video from camera/video

# Stream Video with OpenCV from an Android running IP Webcam (https://play.google.com/store/apps/details?id=com.pas.webcam)
# Code Adopted from http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera

# import urllib2
# import sys
#
#
# host = "10.35.77.247:8080"
# if len(sys.argv)>1:
#     host = sys.argv[1]
#
# hoststr = 'http://' + host + '/video'
# print 'Streaming ' + hoststr
#
# stream=urllib2.urlopen(hoststr)
#
# bytes=''
#
# #create the main figure
# fig = plt.figure()
# plt.ion()
# plt.show()
#
# frame_num = 0
# total_faces=10
# frames = [[]]*total_faces
# while True:
#
#     bytes += stream.read(1024)
#     a = bytes.find('\xff\xd8')
#     b = bytes.find('\xff\xd9')
#     if a != -1 and b != -1:
#         jpg = bytes[a:b + 2]
#         bytes = bytes[b + 2:]
#         frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 1)
#         frames[frame_num%total_faces] = frame
#         frame_num = (frame_num+1)%total_faces
#
#
#         if frame_num == 0 and len(frame[0]):
#
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             net.eval_pics(images=frames)
#             plt.draw()
#             plt.pause(0.001)

while True:
    cap = cv2.VideoCapture('video.mp4')
    #create the main figure
    fig = plt.figure()
    plt.ion()
    plt.show()

    frames=[]
    while(cap.isOpened()):
        for i in range(10):
            ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if len(frames)==6:
            #eval frames
            net.eval_pics(images=frames)
            plt.draw()
            plt.pause(0.001)
            # plt.show(block=False)
            # plt.clf()
            frames=[]


    cap.release()
    cv2.destroyAllWindows()


# for live demo from saved images

# import glob
# image_paths = list(glob.iglob('/home/rocket/PycharmProjects/test/FERPlus/my_src/emo_pics/*'))
# net.eval_pics(image_paths=image_paths)
# plt.show()

#for demo
# img_names=['fer003226'+str(i)+'.png' for i in range(10)]
# net.print_preds(img_names,mode='test')

net.close()
