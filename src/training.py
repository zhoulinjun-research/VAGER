#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from config import *
from vager import *

OUTPUT_DIM = BASE_CLASSES
w = tf.Variable(tf.random_normal([FEATURE_DIM, OUTPUT_DIM]), name='w2')
b = tf.Variable(tf.random_normal([OUTPUT_DIM]), name='b2')
init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'model_500_fixed/model')
    weight = sess.run(w)
    bias = sess.run(b)

avg_base_data = []
for i in range(BASE_CLASSES):
    avg_base_data.append(np.average(readdoc(i, 'base'), axis=0))
avg_base_data = np.array(avg_base_data)
sim = calc_sim(avg_base_data)
# For convenience, we omit the bias term
X, W = training(sim, np.c_[weight.T, bias])

print('Training step ends.')

if not os.path.exists('embedding/'):
    os.mkdir('embedding/')

np.save('embedding/X.npy', X)
np.save('embedding/W.npy', W)

