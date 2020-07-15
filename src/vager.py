#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import math
import random
import time
import sys
import heapq
import scipy.stats as stats
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from config import *
from sklearn import svm
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# Load features
def readdoc(cls, flag='base'):
    if flag == 'base':
        return np.load(BASE_FOLDER + str(cls) + '.npy')
    if flag == 'oneshot':
        return np.load(ONESHOT_FOLDER + str(cls) + '.npy')
    if flag == 'oneshot_origin':
        return np.load(ORIGIN_FOLDER + str(cls) + '.npy')
    if flag == 'val':
        return np.load(BASE_VAL_FOLDER + str(cls) + '.npy')

# Calculate class centroid and covariance matrix
def calc_stats(data):
    mu = []
    cov = []
    for cls, cls_data in enumerate(data):
        print cls
        mu.append(np.average(cls_data, 0))
        cov.append(np.cov(np.transpose(cls_data)))
    return np.array(mu), np.array(cov)

# Calculate correlation between features
def calc_corr(data):
    return np.corrcoef(np.transpose(data))

# Calculate cosine similarity for class centroids
def calc_sim(avg_mat):
    return cosine_similarity(avg_mat, avg_mat)

# Training base classes
def training(sim, params):
    params = params.astype('float32')
    feature_num = len(params[0])
    X = tf.Variable(tf.random_uniform([BASE_CLASSES, HIDDEN_DIM], -1e-2, 1e-2))
    W = tf.Variable(tf.random_uniform([HIDDEN_DIM, feature_num], -1e-2, 1e-2))
    loss1 = tf.reduce_sum(tf.squared_difference(sim, tf.matmul(X, tf.transpose(X))))
    loss2 = tf.reduce_sum(tf.squared_difference(params, tf.matmul(X, W)))
    loss = loss1 + HPARAM * loss2

    optimizer = tf.train.AdamOptimizer(TR_LRATE).minimize(loss)
    init = tf.initialize_all_variables()

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC
    with tf.Session(config=tfconfig) as sess:
        sess.run(init)
        for epoch in range(0, TR_EPOCH_NUM):
            [op, lo, lo1, lo2] = sess.run([optimizer, loss, loss1, loss2])
            if epoch % 100 == 0:
                print 'Episode %d' % epoch
                print 'Total Loss %f, Reconstruction Loss %f, Prediction Loss %f' % (lo, lo1, lo2)
        [op, rX, rW] = sess.run([optimizer, X, W])
    return rX, rW

# Generalization to a new class
def getting_oneshot_params(sim, X, W):
    z = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], -0.01, 0.01))
    loss = tf.reduce_sum(tf.squared_difference(sim, tf.matmul(z, tf.transpose(X))))
    optimizer = tf.train.GradientDescentOptimizer(TE_LRATE).minimize(loss)
    init = tf.initialize_all_variables()

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC
    with tf.Session(config=tfconfig) as sess:
        sess.run(init)
        for epoch in range(TE_EPOCH_NUM):
            [op, lo] = sess.run([optimizer, loss])
        z_res = sess.run(z)
    pre_result = np.dot(z_res, W)
    return np.array(pre_result[0])

# Parameter Refinement (Initializing and Tuning)
def fixed_logi_tuning(data, label, params, method):
    rows, cols = np.shape(params) # Classes, Feature_dim
    W = tf.Variable(params)
    #W = tf.Variable(tf.random_uniform([rows, cols], -1e-4, 1e-4))
    x = tf.placeholder(tf.float32, [None, cols])
    y = tf.placeholder(tf.float32, [None, rows])
    output = tf.matmul(x, tf.transpose(W))
    lamda1 = PRED_LOGI_MODEL
    lamda2 = PRED_LOGI_TRANS
    lamda3 = PRED_LOGI_REG
    if method == 'init':
        lamda2 = 0.0
    loss_model = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))
    loss_trans = tf.reduce_mean(tf.squared_difference(W, params))
    loss_reg = tf.reduce_mean(tf.square(W))
    loss = lamda1 * loss_model + lamda2 * loss_trans + lamda3 * loss_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=PRED_LOGI_RATE).minimize(loss)
    init = tf.initialize_all_variables()

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC
    with tf.Session(config=tfconfig) as sess:
        sess.run(init)
        BATCHNUM = len(data) / PRED_LOGI_BATCH
        for epoch in range(PRED_LOGI_EPOCH):
            lo = 0.0
            lo_m = 0.0
            lo_t = 0.0
            for batch in range(BATCHNUM):
                [op, lo, lo_m, lo_t] = sess.run([optimizer, loss, loss_model, loss_trans], feed_dict={ \
                        x:data[batch * PRED_LOGI_BATCH : (batch+1) * PRED_LOGI_BATCH], \
                        y:label[batch * PRED_LOGI_BATCH : (batch+1) * PRED_LOGI_BATCH]})
            if epoch % 10 == 0:
                print 'Epoch %d: Loss: %f, Model Loss: %f, Trans Loss: %f' % \
                        (epoch, lo, lo_m, lo_t)
        result = sess.run(W)
    return result

# Parameter Refinement (Voting)
def fixed_logi_voting(trdata, tedata, params):
    lamda = VOTING_PARAM
    coef, intercept, acc = logistic_classification(trdata, tedata, ONESHOT_CLASSES, 1)
    return params + lamda * np.c_[coef, intercept]

# Split training and testing in novel classes
def cut_oneshot_data(origin_data, oneshot_data, trnum, tenum):
    oneshot_origin_training =[[] for i in range(len(oneshot_data))]
    oneshot_training = [[] for i in range(len(oneshot_data))]
    oneshot_origin_testing = [[] for i in range(len(oneshot_data))]
    oneshot_testing = [[] for i in range(len(oneshot_data))]

    for ind in range(len(oneshot_data)):
        sample = random.sample(range(len(oneshot_data[ind])), trnum + tenum)
        for s in sample[0:trnum]:
            oneshot_origin_training[ind].append(origin_data[ind][s])
            oneshot_training[ind].append(oneshot_data[ind][s])
        for s in sample[trnum:trnum+tenum]:
            oneshot_origin_testing[ind].append(origin_data[ind][s])
            oneshot_testing[ind].append(oneshot_data[ind][s])
    return np.array(oneshot_training), np.array(oneshot_origin_training), \
            np.array(oneshot_testing), np.array(oneshot_origin_testing)

# Cosine classifier
def cosine_classification(test_data, mu, pred_num=5):
    res = []
    sim = cosine_similarity(test_data, mu)
    for line in sim:
        predict = heapq.nlargest(pred_num, range(len(line)), line.__getitem__)
        res.append(predict)
    return res

# Euclidean classifier
def eucilid_classification(test_data, mu, pred_num=5):
    res = []
    for index, d in enumerate(test_data):
        L = []
        for cls in range(len(mu)):
            tmp = 0.0
            tmp -= np.dot(d - mu[cls], np.transpose(d - mu[cls]))
            L.append(tmp)
        predict = heapq.nlargest(pred_num, range(len(L)), L.__getitem__)
        res.append(predict)
    return res

# Finetuning
def pred_logi_classification(test_data, mu, params, pred_num=1):
    res = []
    for data in test_data:
        value = np.dot(data, params[:, :-1].T) + params[:, -1]
        predict = heapq.nlargest(pred_num, range(len(value)), value.__getitem__)
        res.append(predict)
    return res

# Testing on basic baselines (including cosine, euclid, finetuning)
output = mp.Queue()
def calc_accuracy(cls, data, oneshot_avg, logi_params, method):
    cnt = 0.0
    result = []
    if method == 'cosine':
        result = cosine_classification(data, oneshot_avg)
    elif method == 'eucilid':
        result = eucilid_classification(data, oneshot_avg)
    elif method == 'logi' or method == 'vager':
        result = pred_logi_classification(data, oneshot_avg, logi_params)
    for it in result:
        if cls in it:
            cnt += 1.0
    output.put([cls, cnt / len(result)])

def testing(base_avg, oneshot_data, test_data, X, W, method, fusion, _logiparam=None):
    oneshot_avg = []
    sim = []
    logi_params = []
    logi_params_fixed = []
    finetune_trdata = []
    finetune_label = []
    for cls in range(len(oneshot_data)):
        oneshot_avg.append(np.average(oneshot_data[cls], 0))

    if method == 'vager':
        if _logiparam is None:
            sim = cosine_similarity(oneshot_avg, base_avg)
            for cls in range(len(sim)):
                print('Calculate Parameter for Class %d.' % cls)
                logi_w = getting_oneshot_params(sim[cls], X, W)
                logi_params.append(logi_w)
            logi_params = np.array(logi_params)
        else:
            logi_params = np.copy(_logiparam)
        # Fine-tuning
        classes, shots, features = np.shape(oneshot_data)
        finetune_trdata = np.c_[np.reshape(oneshot_data, [classes*shots, features]), \
                np.ones((classes*shots, 1))]
        finetune_label = np.zeros((classes*shots, classes))
        for cls in range(len(oneshot_data)):
            for shot in range(shots):
                finetune_label[cls*shots+shot][cls] = 1.0
        finetune_trdata, finetune_label = shuffle(finetune_trdata, finetune_label)
        if fusion == 'none':
            logi_params_fixed = logi_params
        elif fusion == 'voting':
            logi_params_fixed = fixed_logi_voting(oneshot_data, test_data, logi_params)
        elif fusion == 'tuning':
            logi_params_fixed = fixed_logi_tuning(finetune_trdata, finetune_label, logi_params, 'tuning')
        elif fusion == 'init':
            logi_params_fixed = fixed_logi_tuning(finetune_trdata, finetune_label, logi_params, 'init')

    # Process Testing
    processes = [mp.Process(target=calc_accuracy, \
            args=(cls, data, oneshot_avg, logi_params_fixed, method)) \
            for cls, data in enumerate(test_data)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results = np.array([output.get() for p in processes])
    #print results
    return [logi_params, np.average(results[:, 1]), results]

# SVM classification
def svm_classification(trdata, tedata):
    tr = []
    tr_label = []
    te = []
    te_label = []
    for cls in range(len(trdata)):
        for data in trdata[cls]:
            tr.append(data)
            tr_label.append(cls)
    for cls in range(len(tedata)):
        for data in tedata[cls]:
            te.append(data)
            te_label.append(cls)
    tr, tr_label = shuffle(tr, tr_label)
    clf = svm.SVC(C=1.0,kernel='linear',probability=True)
    clf.fit(tr, tr_label)
    prob = clf.predict_proba(te)
    acc = np.array([0.0 for i in range(ONESHOT_CLASSES)])
    for index, r in enumerate(prob):
        res = heapq.nlargest(5, range(len(r)), r.__getitem__)
        print te_label[index], res
        if int(te_label[index]) in res:
            acc[te_label[index]] += 1.0
    acc = acc / (len(te_label) / ONESHOT_CLASSES)
    print acc
    return np.average(acc)

# Logistic Regression (Or: Finetuning) for multi-class
def logistic_classification(trdata, tedata, cls_num, return_params=1):
    tr = []
    tr_label = []
    te = []
    te_label = []
    for cls in range(len(trdata)):
        for data in trdata[cls]:
            tr.append(data)
            tr_label.append(cls)
    for cls in range(len(tedata)):
        for data in tedata[cls]:
            te.append(data)
            te_label.append(cls)
    tr, tr_label = shuffle(tr, tr_label)
    clf = LogisticRegression(solver='sag', C=1.0)
    clf.fit(tr, tr_label)
    prob = clf.predict_proba(te)
    acc = np.array([0.0 for i in range(cls_num)])
    for index, r in enumerate(prob):
        res = heapq.nlargest(1, range(len(r)), r.__getitem__)
        if int(te_label[index]) in res:
            acc[te_label[index]] += 1.0
    acc = acc / (len(te_label) / cls_num)
    print acc
    if return_params == 1:
        return clf.coef_, clf.intercept_, np.average(acc)
    else:
        return np.average(acc), acc

# Weighted-LR for multi-class (API1: need to re-calculate similarity matrix)
def weighted_logi(base_avg, oneshot_data, origin_data, test_data, base_params):
    oneshot_avg_origin = []
    logi_param = []
    for cls in range(ONESHOT_CLASSES):
        oneshot_avg_origin.append(np.average(origin_data[cls], 0))
    sim = cosine_similarity(oneshot_avg_origin, base_avg)
    MAXIMUM = 1
    for cls in range(ONESHOT_CLASSES):
        max_index = heapq.nlargest(MAXIMUM, range(len(sim[cls])), sim[cls].__getitem__)
        max_sim = heapq.nlargest(MAXIMUM, sim[cls])
        sim_total = sum(max_sim)
        logi_param.append(np.sum(np.array([max_sim[i] / sim_total * base_params[max_index[i]] for i in range(MAXIMUM)]), 0))

    processes = [mp.Process(target=calc_accuracy, \
            args=(cls, data, None, np.array(logi_param), 'logi')) \
            for cls, data in enumerate(test_data)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results = np.array([output.get() for p in processes])
    #print results
    return np.average(results[:, 1])

# Weighted-LR for multi-class (API2: given similarity matrix)
def weighted_logi_prec(sim, test_data, base_params):
    logi_param = []
    MAXIMUM = 10
    for cls in range(ONESHOT_CLASSES):
        max_index = heapq.nlargest(MAXIMUM, range(len(sim[cls])), sim[cls].__getitem__)
        max_sim = heapq.nlargest(MAXIMUM, sim[cls])
        sim_total = sum(max_sim)
        logi_param.append(np.sum(np.array([max_sim[i] / sim_total * base_params[max_index[i]] for i in range(MAXIMUM)]), 0))

    processes = [mp.Process(target=calc_accuracy, \
            args=(cls, data, None, np.array(logi_param), 'logi')) \
            for cls, data in enumerate(test_data)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results = np.array([output.get() for p in processes])
    #print results
    return np.average(results[:, 1]), results

# Logistic Regression (or: finetuning) for binary case
def binary_logi(base_data, oneshot_data, param):
    tr = []
    tr_label = []
    for data in base_data:
        tr.append(data)
        tr_label.append(0.0)
    for data in oneshot_data:
        tr.append(data)
        tr_label.append(1.0)
    tr, tr_label = shuffle(tr, tr_label)
    clf = LogisticRegression(warm_start=True)
    print param
    if param is not None:
        clf.coef_ = param[:-1]
        clf.intercept_ = param[-1]
    clf.fit(tr, tr_label)
    return np.c_[clf.coef_, clf.intercept_]

# Weighted-LR for binary case (API1)
def binary_weighted_logi_prec(base_avg, oneshot_data, origin_data, base_params):
    oneshot_avg_origin = np.average(origin_data, 0)
    sim = cosine_similarity(oneshot_avg_origin, base_avg)[0]
    MAXIMUM = 10
    max_index = heapq.nlargest(MAXIMUM, range(len(sim)), sim.__getitem__)
    max_sim = heapq.nlargest(MAXIMUM, sim)
    sim_total = sum(max_sim)
    logi_param = np.array([0.0 for i in range(len(base_params[0]))])
    for i in range(MAXIMUM):
        logi_param = logi_param + max_sim[i] / sim_total * base_params[max_index[i]]
    return np.array([logi_param])

def binary_test_acc(tedata, telabel, params):
    output = []
    for i, data in enumerate(tedata):
        value = np.dot(data, params[:, :-1].T) + params[:, -1]
        score = 0.0
        if value > 0:
            score = 1.0 / (1.0 + math.exp(-value))
        else:
            score = 1.0 - 1.0 / (1.0 + math.exp(value))
        output.append(score)
    precision, recall, thres = precision_recall_curve(telabel, output)
    f1 = []
    for i in range(len(precision)):
        if precision[i] + recall[i] < 1e-4:
			f1.append(0.0)
        else:
            f1.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    return [roc_auc_score(telabel, output), max(f1)]

# Parameter Refinement - Voting  for binary case
def binary_voting(w_trans, w_model):
    return w_trans + VOTING_PARAM * w_model

# Parameter Refinement - Tuning for binary case
def binary_tuning(base_data, oneshot_data, w_trans, method):
    training = np.copy(base_data)
    training = np.concatenate((training, np.copy(oneshot_data)), axis=0)
    training = np.c_[training, np.ones(len(training))]
    label = []
    for i in range(len(base_data)):
        label.append([1.0, 0.0])
    for i in range(len(oneshot_data)):
        label.append([0.0, 1.0])
    training, label = shuffle(training, label)
    w_init = np.array([-w_trans[0], w_trans[0]])
    w_new = fixed_logi_tuning(training, label, w_init, method)
    return np.array([w_new[1]])
