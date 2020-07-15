#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import math
import os
import random
import sys
import heapq
from config import *
from vager import *
import tensorflow as tf

if __name__ == '__main__':
    # Testing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC
    sess = tf.Session(config=tfconfig)
    classes = random.sample(range(200), ONESHOT_CLASSES)

    X = np.load('embedding/X.npy')
    W = np.load('embedding/W.npy')
    avg_base_data = []
    for i in range(BASE_CLASSES):
        avg_base_data.append(np.average(readdoc(i, 'base'), axis=0))
    avg_base_data = np.array(avg_base_data)
    print('Finish loading base data.')

    novel_data = []
    novel_data_origin = []
    for c in classes:
        novel_data.append(readdoc(c, 'oneshot'))
        novel_data_origin.append(readdoc(c, 'oneshot_origin'))
    print('Finish loading novel data')

    vager_result = []
    for t in range(TIMES):
        print('Processing experiment group %d' % (t+1))
        trdata, origin_trdata, tedata, origin_tedata = cut_oneshot_data(novel_data_origin, novel_data, SHOTS, TEST_NUM)
        logi_params, avg_result, result = testing(avg_base_data, trdata, tedata, X, W, 'vager', 'voting', None)
        vager_result.append(avg_result)
    vager_result = np.array(vager_result)

    with open(RESULT_FILE, 'w') as f:
        print >>f, 'CLASSES: %s' % classes
        print >>f, 'TIMES: %d' % TIMES
        print >>f, 'SHOTS: %d' % SHOTS
        print >>f, 'FEATURE_DIM: %d' % FEATURE_DIM
        print >>f, 'HIDDEN_DIM: %d' % HIDDEN_DIM
        print >>f, '\n'
        print >>f, 'AVG_VAGER: %s' % np.average(vager_result, axis=0)

    print 'AVG_VAGER: %s' % np.average(vager_result, axis=0)


