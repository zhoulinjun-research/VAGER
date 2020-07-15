#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os

BASE_CLS = 800
ONESHOT_CLS = 200
n_input = 4096
n_output = 500
BASE_FOLDER = '../features/base/'
BASE_VAL_FOLDER = '../features/val/'
ONESHOT_FOLDER = '../features/novel/'
NEW_BASE = '../features/base_500_fixed/'
NEW_BASE_VAL = '../features/val_500_fixed/'
NEW_ONESHOT = '../features/novel_500_fixed/'


if not os.path.exists(NEW_BASE):
    os.mkdir(NEW_BASE)
if not os.path.exists(NEW_BASE_VAL):
    os.mkdir(NEW_BASE_VAL)
if not os.path.exists(NEW_ONESHOT):
    os.mkdir(NEW_ONESHOT)

w1 = tf.Variable(tf.random_normal([n_input, n_output]), name='w1')
b1 = tf.Variable(tf.random_normal([n_output]), name='b1')
init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'model_500_fixed/model')
    w1_val = sess.run(w1)
    b1_val = sess.run(b1)
    for cls in range(BASE_CLS):
        data = np.load(BASE_FOLDER + str(cls) + '.npy')
        new = np.dot(data, w1_val) + b1_val
        np.save(NEW_BASE + str(cls) + '.npy', new)
    for cls in range(BASE_CLS):
        data = np.load(BASE_VAL_FOLDER + str(cls) + '.npy')
        new = np.dot(data, w1_val) + b1_val
        np.save(NEW_BASE_VAL + str(cls) + '.npy', new)
    for cls in range(ONESHOT_CLS):
        data = np.load(ONESHOT_FOLDER + str(cls) + '.npy')
        new = np.dot(data, w1_val) + b1_val
        np.save(NEW_ONESHOT + str(cls) + '.npy', new)

