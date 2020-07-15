#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import random
from sklearn.utils import shuffle

CLS_NUM = 800
TESTING_NUM = 50
training_data = []
training_label = []
testing_data = []
testing_label = []
A = np.eye(CLS_NUM)
for cls in range(CLS_NUM):
    data = np.load('../features/base/'+str(cls)+'.npy')
    test_index = random.sample(range(len(data)), 50)
    train_index = list(set(range(len(data))).difference(set(test_index)))
    for i in train_index:
        training_data.append(data[i])
        training_label.append(A[cls])
    for i in test_index:
        testing_data.append(data[i])
        testing_label.append(A[cls])
training_data = np.array(training_data)
training_label = np.array(training_label)
testing_data = np.array(testing_data)
testing_label = np.array(testing_label)
_training_data, _training_label = shuffle(training_data, training_label)
_testing_data, _testing_label = shuffle(testing_data, testing_label)
np.save('../features/original/train.npy', _training_data)
np.save('../features/original/train_label.npy', _training_label)
np.save('../features/original/test.npy', _testing_data)
np.save('../features/original/test_label.npy', _testing_label)


