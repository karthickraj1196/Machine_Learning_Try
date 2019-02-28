# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:55:35 2019

@author: pushparajkarthick_d
"""

import numpy as np
import sklearn
import pickle
from sklearn.neural_network import MLPClassifier

xs = np.array([0,0,0,1,1,0,1,1],dtype=int).reshape(4,2)
ys = np.array([0,1,1,0],dtype=int).reshape(4,)

model = sklearn.neural_network.MLPClassifier(activation='relu', max_iter=10000, hidden_layer_sizes=(4,2))
model.fit(xs, ys)

filename = 'xor_model.pkl'
pickle.dump(model, open(filename, 'wb'))

            
