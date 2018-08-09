import tensorflow as tf


def logits_compute(embeddings, label_batch, embedding_size, num_classes):
    m = 0.5
    s = 64
    with tf.name_scope('AM_logits'):
        kernel = tf.Variable(tf.truncated_normal([embedding_size, num_classes]))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(embeddings, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        phi = cos_theta - m
        label_onehot = tf.one_hot(label_batch, num_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)

        return adjust_theta
