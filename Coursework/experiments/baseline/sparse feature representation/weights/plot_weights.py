import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle
def load_from_file(filename):
    """ Load object from file
    """
    object = []
    f = open(filename + '.pckl', 'rb')
    object = pickle.load(f)
    f.close()
    return object
def save_to_file(filename, object):
    """ Save object to file
    """
    f = open(filename + '.pckl', 'wb')
    pickle.dump(object, f)
    f.close()

hidden1_weights_L1 = load_from_file('hidden1_weights_L1')
hidden2_weights_L1 = load_from_file('hidden2_weights_L1')
hidden1_weights_L2 = load_from_file('hidden1_weights_L2')
hidden2_weights_L2 = load_from_file('hidden2_weights_L2')

interval=[-0.5, 0.5]

"""Plots a normalised histogram of an array of parameter values."""
fig_1 = plt.figure(figsize=(10, 8))
ax_1 = fig_1.add_subplot(2, 2, 1)
ax_2 = fig_1.add_subplot(2, 2, 2)
ax_3 = fig_1.add_subplot(2, 2, 3)
ax_4 = fig_1.add_subplot(2, 2, 4)

ax_1.hist(hidden1_weights_L1, 50, interval, normed=True)
ax_1.set_xlabel('Weights conv layer 1 L1=0.001', fontsize=14)
ax_1.set_ylabel('Normalised frequency density', fontsize=14)

ax_2.hist(hidden2_weights_L1, 50, interval, normed=True)
ax_2.set_xlabel('Weights conv layer 2 L1=0.001', fontsize=14)
ax_2.set_ylabel('Normalised frequency density', fontsize=14)

ax_3.hist(hidden1_weights_L2, 50, interval, normed=True)
ax_3.set_xlabel('Weights conv layer 1 L2=0.01', fontsize=14)
ax_3.set_ylabel('Normalised frequency density', fontsize=14)

ax_4.hist(hidden2_weights_L2, 50, interval, normed=True)
ax_4.set_xlabel('Weights conv layer 2 L2=0.01', fontsize=14)
ax_4.set_ylabel('Normalised frequency density', fontsize=14)

fig_1.savefig('weightshistogram.pdf')
