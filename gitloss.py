import tensorflow as tf
import numpy as np

lambda_c = 1.
lambda_g = 1.


def get_git_loss(embeddings, labels, num_classes):
    centers = tf.get_variable(name='centers', shape=[num_classes, embeddings.get_shape()[1]], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    loss_c = tf.reduce_mean(tf.square(embeddings - centers_batch))

    diffs = (embeddings[:, tf.newaxis] - num_classes[tf.newaxis, :])
    diffs_shape = tf.shape(diffs)
    print(diffs.get_shape())

    mask = 1 - tf.eye(diffs_shape[0], diffs_shape[1], dtype=diffs.dtype)
    diffs = diffs * mask[:, :, tf.newaxis]

    loss_g = tf.reduce_mean(tf.divide(1, 1 + tf.square(diffs)))

    diff = centers_batch - embeddings
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_label, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast(1 + appear_times, dtype=tf.float32))
    diff = 0.5 * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    loss = lambda_c * loss_c + lambda_g * loss_g

    return loss, centers_update_op



