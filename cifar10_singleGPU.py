# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Distributed MNIST training and validation, with model replicas.
A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.
The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import sys
import time
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf

import cifar10_input
from cifar10_input import *
from resnet import resnet, resnet_loss
from cnn import cnn, cnn_loss
from mlp import mlp, mlp_loss


flags = tf.app.flags
flags.DEFINE_string("data_dir", "./cifar10-data",
                    "Directory for storing cifar-10 data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("train_steps", 100000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("log_interval", 500,
                     "Interval number of (local) training steps to log")
flags.DEFINE_integer("test_interval", 2000,
                     "Number of (local) training steps to perform testing.")
flags.DEFINE_string("model", "resnet",
                    "model name, candidate list: [resnet, cnn, mlp].")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
flags.DEFINE_integer("num_blocks", 18,
                     "Number of blocks of networks.")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
                           time''')
tf.app.flags.DEFINE_integer('decay_step', 20000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_string('optim', 'momentum',
                           'Optimizer to be used, candidate list: [adam, momentum, gd]')

FLAGS = flags.FLAGS


def main(unused_argv):
    maybe_download_and_extract(FLAGS.data_dir)
    if FLAGS.download_only:
        sys.exit(0)

    # Check model.
    if FLAGS.model == "resnet":
        inference, loss = resnet, resnet_loss
        FLAGS.learning_rate = 0.1
        FLAGS.optim = "momentum"
        FLAGS.num_blocks = 18
    elif FLAGS.model == "cnn":
        inference, loss = cnn, cnn_loss
        FLAGS.learning_rate = 0.01
        FLAGS.optim = "adam"
    elif FLAGS.model == "mlp":
        inference, loss = mlp, mlp_loss
        FLAGS.learning_rate = 0.01
        FLAGS.optim = "adam"
    else:
        assert "Model Type Error !"

    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
    y_ = tf.placeholder(tf.int64, [None])
    lr_placeholder = tf.placeholder(tf.float32)
    # batch_size_ = tf.placeholder(tf.int64, name="batch_size_")

    logits = inference(x, FLAGS.num_blocks)

    y = tf.nn.softmax(logits)

    labels = tf.cast(y_, tf.int64)

    total_loss = loss(logits, labels)

    correct_prediction = tf.equal(tf.arg_max(y, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        FLAGS.decay_step,
        FLAGS.lr_decay_factor,
        staircase=True)

    if FLAGS.optim == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif FLAGS.optim == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optim == 'gd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        assert "Optimizer Type Error !"

    train_step = opt.minimize(total_loss, global_step=global_step)

    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # Load Data
        all_data, all_labels = prepare_train_data(FLAGS.data_dir)
        valid_data, valid_labels = read_validation_data(FLAGS.data_dir)

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        def valid(local_step):
            batch_size = FLAGS.batch_size
            n_test = valid_data.shape[0]

            # Validation test
            current_step = 0
            cumulated_acc = 0.0
            while current_step * batch_size < n_test:
                current_num = min(batch_size, n_test - current_step * batch_size)
                batch_x = valid_data[current_step * batch_size:current_step * batch_size + current_num, :, :, :]
                batch_y = valid_labels[current_step * batch_size:current_step * batch_size + current_num]
                valid_feed = {x: batch_x, y_: batch_y, lr_placeholder: FLAGS.learning_rate}

                acc = sess.run([accuracy], feed_dict=valid_feed)
                cumulated_acc += acc[0] * current_num

                current_step += 1

            cumulated_acc /= n_test

            print("After %d training step(s), prediction accuracy = %g, time cost = %f" %
                  (local_step, cumulated_acc, time.time() - time_begin))

            return cumulated_acc

        local_step = 0
        best_valid_acc = 0.0
        interval_cumulated_acc = 0.0
        while True:
            batch_x, batch_y = generate_augment_train_batch(all_data, all_labels, FLAGS.batch_size)
            train_feed = {x: batch_x, y_: batch_y, lr_placeholder: FLAGS.learning_rate}

            _, step, acc = sess.run([train_step, global_step, accuracy], feed_dict=train_feed)

            interval_cumulated_acc += acc
            local_step += 1

            now = time.time()
            if local_step % FLAGS.log_interval == 0:
                print("%f: Acc: %g training step %d done (global step: %d)" %
                      (now, interval_cumulated_acc / FLAGS.log_interval, local_step, step))
                interval_cumulated_acc = 0.0

            if local_step % FLAGS.test_interval == 0:
                best_valid_acc = max(best_valid_acc, valid(step))

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)
        print("Best Validation Acc: %g" % best_valid_acc)


if __name__ == "__main__":
    tf.app.run()
