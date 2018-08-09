import numpy as np
import facenet
import lfw
import os
import time
import math
import tensorflow as tf
from scipy import misc


def evaluate_double(sess, phase_train_placeholder, embeddings, lfw_batch_size, step, summary_writer, best_accuracy,
                    images_placeholder, lfw_dir, lfw_pairs, lfw_file_ext, saver, model_dir, model_name):
    start_time = time.time()

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)
    batch_size = lfw_batch_size
    nrof_images = len(paths)
    print('Validate on %d images' % nrof_images)
    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, 512))
    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = load_data(paths_batch)
        images_flip = np.flip(images, 2)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        feed_dict_flip = {images_placeholder: images_flip, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        emb_flip = sess.run(embeddings, feed_dict=feed_dict_flip)
        emb_average = (emb + emb_flip) / 2.0
        emb_array[start_index:end_index, :] = emb_average

    accuracy, thre = evaluate_with_no_cv(emb_array, actual_issame)

    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)
        save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step)

    print('Accuracy: %1.3f Threshold: %1.3f' % (accuracy, thre))

    lfw_time = time.time() - start_time
    summary = tf.Summary()
    summary.value.add(tag='lfw/accuracy', simple_value=accuracy)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)

    return best_accuracy


def evaluate_with_no_cv(emb_array, actual_issame):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]

    nrof_thresholds = len(thresholds)
    accuracys = np.zeros((nrof_thresholds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for threshold_idx, threshold in enumerate(thresholds):
        _, _, accuracys[threshold_idx] = facenet.calculate_accuracy(threshold, dist, actual_issame)

    best_acc = np.max(accuracys)
    best_thre = thresholds[np.argmax(accuracys)]
    return best_acc, best_thre


def load_data(image_paths):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 112, 96, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = (img * 1.0 - 127.5) / 128
        images[i, :, :, :] = img
    return images


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
