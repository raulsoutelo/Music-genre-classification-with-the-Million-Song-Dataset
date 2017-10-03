import os
import time
#import tensorflow as tf
import numpy as np
#from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider
import matplotlib.pyplot as plt

#function found in the Internet to store our variables on files
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

err_val_L2_in_all_no_output = load_from_file('err_val_L2_in_all_no_output')
acc_val_L2_in_all_no_output = load_from_file('acc_val_L2_in_all_no_output')
err_val_no_L1 = load_from_file('err_val_no_L1')
acc_val_no_L1 = load_from_file('acc_val_no_L1')
err_val_L1_0_0001 = load_from_file('err_val_L1_0_0001')
acc_val_L1_0_0001 = load_from_file('acc_val_L1_0_0001')
err_val_L1_0_001 = load_from_file('err_val_L1_0_001')
acc_val_L1_0_001 = load_from_file('acc_val_L1_0_001')
err_val_L1_0_01 = load_from_file('err_val_L1_0_01')
acc_val_L1_0_01 = load_from_file('acc_val_L1_0_01')

# Plot the change in the validation and training set error over training.
fig_1 = plt.figure(figsize=(12, 6))
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_2 = fig_1.add_subplot(1, 2, 2)

l = sorted(err_val_L1_0_0001.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val_L1=0.0001')
l = sorted(acc_val_L1_0_0001.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val_L1=0.0001')

l = sorted(err_val_L1_0_001.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val_L1=0.001')
l = sorted(acc_val_L1_0_001.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val_L1=0.001')

l = sorted(err_val_L1_0_01.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val_L1=0.01')
l = sorted(acc_val_L1_0_01.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val_L1=0.01')

l = sorted(err_val_no_L1.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val_no_reg')
l = sorted(acc_val_no_L1.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val_no_reg')

l = sorted(err_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val_L2=0.01')
l = sorted(acc_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val_L2=0.01')

ax_1.legend(loc=0,prop={'size':12})
ax_1.set_xlabel('Epoch number')
ax_2.legend(loc=0,prop={'size':12})
ax_2.set_xlabel('Epoch number')

fig_1.savefig('l1regularization.pdf')
