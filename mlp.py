import math
import tensorflow as tf

from cifar10_input import *

HIDDEN_UNITS = 200


def mlp(x, block_num=None):
    x = tf.reshape(x, [-1, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    # Variables of the hidden layer
    hid_w = tf.get_variable("hid_w", [IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH, HIDDEN_UNITS],
                            initializer=tf.truncated_normal_initializer(stddev=1.0 / IMG_WIDTH, dtype=tf.float32), dtype=tf.float32)
    hid_b = tf.get_variable("hid_b", [HIDDEN_UNITS])

    # Variables of the softmax layer
    sm_w = tf.get_variable("sm_w", [HIDDEN_UNITS, 10],
                           initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(HIDDEN_UNITS), dtype=tf.float32), dtype=tf.float32)
    sm_b = tf.get_variable("sm_b", [10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    #y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    y = tf.nn.xw_plus_b(hid, sm_w, sm_b)

    return y


def mlp_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def mlp_tower_loss(logits, labels, scope):
    _ = mlp_loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss
