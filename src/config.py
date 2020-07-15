#!/usr/bin/env python
# encoding: utf-8

# Folder containing base classes feature
BASE_FOLDER = '../features/base_500_fixed/'
# Folder containing novel classes feature (after deducting feature dimension by ddim.py)
ONESHOT_FOLDER = '../features/novel_500_fixed/'
# Folder containing novel classes feature (original 4096-dim feature)
ORIGIN_FOLDER ='../features/novel/'
# Folder containing validation dataset in Imagenet for base classes
BASE_VAL_FOLDER = '../features/val_500_fixed/'

# Set result file
RESULT_FILE_NAME = 'result'
RESULT_FILE = '../result/' + RESULT_FILE_NAME + '.txt'
SUPP_RESULT_FILE = '../result/' + RESULT_FILE_NAME + '_supp.txt'

# GPU settings
GPU_FRAC = 0.4
# test group number
TIMES = 10
# shot number
SHOTS = 1
# test images number per class for each test group
TEST_NUM = 100

# LR settings
# base classes number
BASE_CLASSES = 800
# oneshot classes number
ONESHOT_CLASSES = 10

# feature dimension
FEATURE_DIM = 500
# embedding layer dimension
HIDDEN_DIM = 600

# balance hyperparameter for regression and reconstruction for visual analogy graph
HPARAM = 1.0
# hyperparameter for parameter refinement (initializing and tuning)
PRED_LOGI_MODEL = 1.0
PRED_LOGI_TRANS = 1.0 # only for tuning
PRED_LOGI_REG = 0.0 # only for initializing
# hyperparameter for parameter refinement (voting)
VOTING_PARAM = 0.2 # TransLoss + param * ModelLoss

# Learning rate for training phase
TR_LRATE = 2e-4
# Learning rate for generalization phase
TE_LRATE = 1e-3
# Learning rate for parameter refinement
PRED_LOGI_RATE = 0.05

# Epochs for training, generalization and parameter refinement
TR_EPOCH_NUM = 50000
TE_EPOCH_NUM = 10000
PRED_LOGI_EPOCH = 20

# batch number for parameter refinement
# noticing we use oneshot_class for 200 in paper, there's no need to worry about reminders.
# or you will change corresponding code in vager.py or use a suitable batch number
PRED_LOGI_BATCH = 100




