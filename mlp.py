import math
import tensorflow as tf

from cifar10_input import *

HIDDEN_UNITS = 200


def mlp(x, block_num=None):
    x = tf.reshape(x, [-1, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH, HIDDEN_UNITS],
            stddev=1.0 / IMG_WIDTH),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([HIDDEN_UNITS]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [HIDDEN_UNITS, 10],
            stddev=1.0 / math.sqrt(HIDDEN_UNITS)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    #y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    y = tf.nn.xw_plus_b(hid, sm_w, sm_b)

    return y


def mlp_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    return cross_entropy