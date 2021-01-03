import numpy as np


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def eval_acc(threshold, diff):
    y_pred = diff[:, 0] > threshold
    y_true = diff[:, 1]
    acc = 1.0 * (y_true == y_pred).sum() / len(y_true)
    return acc

def find_best_threshold(thresholds, predicts):
    best_thr = best_acc = 0
    for thr in thresholds:
        acc = eval_acc(thr, predicts)
        if acc >= best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr