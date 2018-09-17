import numpy as np
import facenet
import lfw
import os
import time
import math
import tensorflow as tf
from scipy import misc


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder, embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir,
             step, summary_writer, best_accuracy, saver_save, model_dir, model_name):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0, len(image_paths)), 1)
    image_paths_array = np.expand_dims(np.array(image_paths), 1)

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame) * 2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb

    assert np.array_equal(lab_array, np.arange(
        nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    if np.mean(accuracy) > best_accuracy:
        save_variables_and_metagraph(sess, saver_save, summary_writer, model_dir, model_name, step)
        best_accuracy = np.mean(accuracy)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
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


def load_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size[0], image_size[1], 3))
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
