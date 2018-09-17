import tensorflow as tf
import math


def logits_compute(embeddings, labels, embedding_size, num_classes, batch_size):
    s = 64
    sphere_m = 0.9
    cosine_m = 0.15
    arc_m = 0.4
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    with tf.variable_scope('arcface_loss'):
        weights = tf.get_variable(name='embedding_weights', shape=(embedding_size, num_classes), initializer=w_init,
                                  dtype=tf.float32)
        weights_unit = tf.nn.l2_normalize(weights, axis=0)
        cos_t = tf.matmul(embeddings, weights_unit)
        print('cos_t:', cos_t.get_shape())
        ordinal = tf.constant(list(range(0, batch_size)), tf.int64)
        ordinal_y = tf.stack([ordinal, labels], axis=1)
        zy = cos_t * s
        print('zy:', zy.get_shape())
        sel_cos_t = tf.gather_nd(zy, ordinal_y)

        cos_value = sel_cos_t / s
        t = tf.acos(cos_value)
        t = sphere_m * t + arc_m
        body = tf.cos(t)
        body = body - cosine_m
        new_zy = body * s
        print('new_zy:', new_zy.get_shape())
        updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t),
                                                  shape=(batch_size, zy.get_shape().as_list()[-1])),
                                name='updated_logits')
        print('updated logits', updated_logits.get_shape())

    return updated_logits


def logits_compute_2(embedding, labels, embedding_size, out_num):
    s = 64
    margin_a = 1.
    margin_b = 0.2
    margin_m = 0.3
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    weights = tf.get_variable(name='embedding_weights', shape=(embedding_size, out_num),
                              initializer=w_init, dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    ordinal = tf.constant(list(range(0, embedding.get_shape().as_list()[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
        if margin_a == 1.0 and margin_m == 0.0:
            s_m = s * margin_b
            new_zy = sel_cos_t - s_m
        else:
            cos_value = sel_cos_t / s
            t = tf.acos(cos_value)
            if margin_a != 1.0:
                t = t * margin_a
            if margin_m > 0.0:
                t = t + margin_m
            body = tf.cos(t)
            if margin_b > 0.0:
                body = body - margin_b
            new_zy = body * s
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    return updated_logits
