import os
import sys
import random
import argparse
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops, array_ops
import tensorflow.contrib.slim as slim
import numpy as np

import facenet
import lfw
import models.L_Resnet50E_IR as network
from lfw_eval import evaluate
# from add_loss import logits_compute
# from arc_loss import logits_compute
from combined_margin_arc_loss import logits_compute


lfw_pairs = '../lfw_aligned/pairs.txt'
lfw_batch_size = 100
image_size = (112, 96)
embedding_size = 512

random_seed = 666


def main(args):
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    np.random.seed(seed=random_seed)
    random.seed(random_seed)
    train_set = facenet.get_dataset(args.data_dir)
    num_classes = len(train_set)
    cur_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    print('LFW directory: %s' % args.lfw_dir)
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
    lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, 'jpg')

    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)
        global_step = tf.Variable(0, trainable=False)
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=5822700, dtypes=[tf.string, tf.int64], shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        num_preprocess_threads = 4
        images_and_labels = []
        for _ in range(num_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.cast(tf.image.decode_image(file_contents, channels=3), tf.float32)
                image = tf.image.random_flip_left_right(image)
                image.set_shape((image_size[0], image_size[1], 3))
                images.append(tf.subtract(image, 127.5) * 0.0078125)
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size_placeholder,
                                                       shapes=[(image_size[0], image_size[1], 3), ()],
                                                       enqueue_many=True,
                                                       capacity=4 * num_preprocess_threads * args.batch_size,
                                                       allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % num_classes)
        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, 0.8, phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=embedding_size, weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # embeddings = tf.identity(prelogits, 'embeddings')
        logits = logits_compute(embeddings, label_batch, embedding_size, num_classes, args.batch_size)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, args.epoch_size, 1,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        for weights in slim.get_variables_by_name('kernel'):
            kernel_regularization = tf.contrib.layers.l2_regularizer(args.weight_decay)(weights)
            print(weights)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, kernel_regularization)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        if args.weight_decay == 0:
            total_loss = tf.add_n([cross_entropy_mean], name='total_loss')
        else:
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        tf.add_to_collection('losses', total_loss)

        loader = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        # train_op = facenet.train(total_loss, global_step, args.optimizer,
        #    learning_rate, args.moving_average_decay, tf.trainable_variables(), args.log_histograms)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step,
        #                                                           var_list=tf.trainable_variables())
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss, global_step=global_step,
                                                                                    var_list=tf.trainable_variables())
        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained:
                print('Restoring model: %s' % args.pretrained)
                loader.restore(sess, args.pretrained)
            print('Running training')
            epoch = 0
            best_accuracy = 0.0

            while epoch < args.num_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
                      labels_placeholder, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                      global_step, total_loss, train_op, summary_op, summary_writer, regularization_losses, saver,
                      args.train_dir, cur_time)

                print('validation running...')
                # accuracy = evaluate_double(sess, phase_train_placeholder, embeddings, lfw_batch_size, step,
                #                            summary_writer, image_batch, args.lfw_dir, lfw_pairs, 'jpg', image_size)

                best_accuracy = evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder,
                                         phase_train_placeholder, batch_size_placeholder, embeddings, label_batch,
                                         lfw_paths, actual_issame, lfw_batch_size, 10, args.log_dir, step,
                                         summary_writer, best_accuracy, saver, args.train_dir, cur_time)


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, saver, model_dir, cur_time):
    batch_number = 0

    lr = args.learning_rate

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    print('training a epoch...')
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if batch_number % 100 == 0:
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag='train/loss', simple_value=err)
            summary.value.add(tag='train/regloss', simple_value=np.sum(reg_loss))
            summary_writer.add_summary(summary, global_step=step)
            if batch_number % 1000 == 0 and batch_number:
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, cur_time, step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    print('Saving variables')
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        saver.export_meta_graph(metagraph_filename)
    summary = tf.Summary()
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='.', default='../faces_emore_aligned_3')
    parser.add_argument('--batch_size', type=int, help='.', default=256)
    parser.add_argument('--epoch_size', type=int, help='.', default=1000)
    parser.add_argument('--learning_rate', type=float, help='.', default=0.1)
    parser.add_argument('--weight_decay', type=float, help='.', default=0.0)
    parser.add_argument('--num_epochs', type=int, help='.', default=100)
    parser.add_argument('--train_dir', type=str, help='.', default='./trained_models/arc')
    parser.add_argument('--log_dir', type=str, help='.', default='./logs/log')
    parser.add_argument('--lfw_dir', type=str, help='.', default='../lfw_aligned')
    parser.add_argument('--pretrained', type=str, help='.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
