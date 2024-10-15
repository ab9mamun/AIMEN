#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from . import common


def train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest):

    clf.fit(Xtrain, Ytrain)
    Y_pred_plain = clf.predict(Xtest)
    #Y_pred_plain = np.argmax(Y_pred, axis=1)
    Y_plain = Ytest
    common.print_metrics(Y_plain, Y_pred_plain)


def train_and_test_3NN(Xtrain, Ytrain, Xtest, Ytest):
    clf = KNeighborsClassifier(n_neighbors=3)
    train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest)


def train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest):
    clf = KNeighborsClassifier(n_neighbors=1)
    train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest)


def train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest):
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest)


def train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest):
    clf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1)
    clf.fit(Xtrain, Ytrain)

    Y_plain = Ytest
    Y_pred_plain = clf.predict(Xtest)

    common.print_metrics(Y_plain, Y_pred_plain)