#!/usr/bin/env python
# encoding: utf-8


"""
@author: Jackling Gu
@file: gen_small_data.py
@time: 17-9-21 11:23
"""
from keras.datasets import cifar10
import pickle

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

res = (x_train[:2000], y_train[:2000])

fout = open("data/cifar_data.pkl", 'wb')
pickle.dump(res, fout)
