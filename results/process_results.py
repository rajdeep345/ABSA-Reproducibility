import os
import sys

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

path = sys.argv[1]
if len(sys.argv) == 3:
    hard_set = sys.argv[2]
else:
    hard_set = 'False'

fs = os.listdir(path)

def parse_fname(fname):
    parts = fname.split('-')
    model = parts[0]
    train = parts[1]
    test = parts[2]
    try:
        acc = 100*float(parts[-3])
        f1 = 100*float(parts[-1][:-4])
        return model, train, test, acc, f1
    except:
        return

def parse_cdomain(fname):
    parts = fname.split('-')
    model = parts[0]
    train = parts[1]
    test = parts[2]
    fpath = os.path.join('cross_domain', fname)
    try:
        df = pd.read_csv(fpath)
    except:
        print(fname)
    test_acc = 100*list(df['test_acc'])[-1]
    test_f1 = 100*list(df['test_f1'])[-1]
    return model, train, test, test_acc, test_f1

def parse_hardset(fname):
    parts = fname.split('-')
    model = parts[0]
    train = parts[1]
    test = parts[2]
    hard_or_not = test + '_hard_idxs.txt'
    hard_idxs = np.genfromtxt(hard_or_not)
    true_labels_path = os.path.join('in_domain', test + '-target.csv')
    y_true = np.genfromtxt(true_labels_path)
    pred_labels_path = os.path.join('in_domain', fname)
    y_pred = np.genfromtxt(pred_labels_path)
    y_pred = np.argmax(y_pred, axis=1)

    true_labels = []
    pred_labels = []
    
    for (true, pred, hard) in zip(y_true, y_pred, hard_idxs):
        if hard == 1:
            true_labels.append(true)
            pred_labels.append(pred)
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    test_acc = 100*accuracy_score(true_labels, pred_labels)
    
    test_f1 = 100*precision_recall_fscore_support(true_labels, pred_labels, average='macro')[2]
    return model, train, test, test_acc, test_f1


model_acc = defaultdict(list)
model_f1 = defaultdict(list)

for f in fs:
    if 'target.csv' in f:
        continue
    if hard_set == 'True':
        model, train, test, acc, f1 = parse_hardset(f)
    elif path == 'in_domain' or 'cross_domain_incremental':
        model, train, test, acc, f1 = parse_fname(f)
    elif path == 'cross_domain':
        model, train, test, acc, f1 = parse_cdomain(f)
    elif path == 'contrast_logs':
        model, train, test, acc, f1 = parse_fname(f)
    else:
        continue
    if path == 'contrast_logs':
        if train == test:
            continue
    model_acc[(model, train, test)].append(acc)
    model_f1[(model, train, test)].append(f1)

for mdset in model_acc:
    model, train, test = mdset
    accs = model_acc[mdset]
    f1s = model_f1[mdset]
    mu_acc = np.mean(accs)
    std_acc = np.std(accs)
    mu_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    print(f"{model},{train},{test},{mu_acc},{std_acc},{mu_f1},{std_f1}")
