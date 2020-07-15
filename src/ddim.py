#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os

learning_rate = 0.001
HPARAM = 0.1
epochs = 50
training_batch_size = 200
testing_batch_size = 200
display_step = 20
testing_step = 500

n_input = 4096
n_classes = 800
n_output = 500

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def getting_data():
    trdata = np.load('../features/original/train.npy')
    trlabel = np.load('../features/original/train_label.npy')
    tedata = np.load('../features/original/test.npy')
    telabel = np.load('../features/original/test_label.npy')
    return trdata, trlabel, tedata, telabel

def perceptron(X, weights, bias):
    vec = tf.matmul(X, weights['w1']) + bias['b1']
    relu_vec = tf.nn.relu(vec)
    result = tf.matmul(relu_vec, weights['w2']) + bias['b2']
    return vec, result

weights = {
        'w1': tf.Variable(tf.random_normal([n_input, n_output], stddev=0.003), name='w1'),
        'w2': tf.Variable(tf.random_normal([n_output, n_classes], stddev=0.003), name='w2')
        }
bias = {
        'b1': tf.Variable(tf.random_normal([n_output], stddev=0.003), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_classes], stddev=0.003), name='b2')
        }

output, pred = perceptron(x, weights, bias)
cost_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost_regular = tf.reduce_sum(tf.square(weights['w1']))
cost = cost_model + HPARAM * cost_regular
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
trdata, trlabel, tedata, telabel = getting_data()
saver = tf.train.Saver({'w1':weights['w1'], 'b1':bias['b1'], 'w2':weights['w2'], 'b2':bias['b2']})
with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    trbatches = len(trdata) / training_batch_size
    tebatches = len(tedata) / testing_batch_size
    for e in range(epochs):
        for batch in range(trbatches):
            batch_xs = trdata[batch * training_batch_size: (batch+1) * training_batch_size]
            batch_ys = trlabel[batch * training_batch_size: (batch+1) * training_batch_size]
            [c, cm, cr, a, op, prd] = sess.run([cost, cost_model, cost_regular, accuracy, optimizer, pred], feed_dict={x: batch_xs, y: batch_ys})
            if batch % display_step == 0:
                print 'Epoch %d, Batch %d: Loss: %f, Model Loss: %f, Regular Loss: %f, Accuracy: %f' % (e, batch, c, cm, cr, a)

            if batch % testing_step == 0:
                print 'Testing...'
                acc = 0.0
                loss = 0.0
                for tebatch in range(tebatches):
                    te_batch_xs = tedata[tebatch * testing_batch_size: (tebatch+1) * testing_batch_size]
                    te_batch_ys = telabel[tebatch * testing_batch_size: (tebatch+1) * testing_batch_size]
                    acc += sess.run(accuracy, feed_dict={x: te_batch_xs, y: te_batch_ys})
                    loss += sess.run(cost, feed_dict={x: te_batch_xs, y:te_batch_ys})
                acc /= tebatches
                loss /= tebatches
                print 'Accuracy: %f, Loss: %f' % (acc, loss)
    os.mkdir('model_'+str(n_output)+'_fixed')
    saver.save(sess, 'model_'+str(n_output)+'_fixed/model')
