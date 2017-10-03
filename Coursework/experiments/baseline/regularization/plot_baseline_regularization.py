import os
import time
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


err_val_L2_in_all = load_from_file('err_val_L2_in_all')
acc_val_L2_in_all = load_from_file('acc_val_L2_in_all')
err_val_L2_in_all_no_output = load_from_file('err_val_L2_in_all_no_output')
acc_val_L2_in_all_no_output = load_from_file('acc_val_L2_in_all_no_output')
err_val_L2_in_FC = load_from_file('err_val_L2_in_FC')
acc_val_L2_in_FC = load_from_file('acc_val_L2_in_FC')

# Plot the change in the validation and training set error over training.
fig_1 = plt.figure(figsize=(12, 6))
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_2 = fig_1.add_subplot(1, 2, 2)

l = sorted(err_val_L2_in_FC.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val no reg conv layers')
l = sorted(acc_val_L2_in_FC.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val no reg conv layers')

l = sorted(err_val_L2_in_all.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val L2=0.01 all layers')
l = sorted(acc_val_L2_in_all.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val L2=0.01 all layers')

l = sorted(err_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val no reg output layer')

l = sorted(acc_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val no reg output layer')

ax_1.legend(loc=0,prop={'size':12})
ax_1.set_xlabel('Epoch number')
ax_2.legend(loc=0,prop={'size':12})
ax_2.set_xlabel('Epoch number')

fig_1.savefig('baselinereg.pdf')
