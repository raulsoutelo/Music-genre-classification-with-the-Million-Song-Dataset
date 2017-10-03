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

err_val_ConvTime_KernelHeight_5 = load_from_file('err_val_ConvTime_KernelHeight_5')
acc_val_ConvTime_KernelHeight_5 = load_from_file('acc_val_ConvTime_KernelHeight_5')

err_val_ConvTime_KernelHeight_3 = load_from_file('err_val_ConvTime_KernelHeight_3')
acc_val_ConvTime_KernelHeight_3 = load_from_file('acc_val_ConvTime_KernelHeight_3')

err_val_ConvTime_KernelHeight_10 = load_from_file('err_val_ConvTime_KernelHeight_10')
acc_val_ConvTime_KernelHeight_10 = load_from_file('acc_val_ConvTime_KernelHeight_10')

err_val_L2_in_all_no_output = load_from_file('err_val_L2_in_all_no_output')
acc_val_L2_in_all_no_output = load_from_file('acc_val_L2_in_all_no_output')

# Plot the change in the validation and training set error over training.
fig_1 = plt.figure(figsize=(12, 6))
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_2 = fig_1.add_subplot(1, 2, 2)

l = sorted(err_val_ConvTime_KernelHeight_5.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val kernel height = 5')
l = sorted(acc_val_ConvTime_KernelHeight_5.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val kernel height = 5')

l = sorted(err_val_ConvTime_KernelHeight_3.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val kernel height = 3')
l = sorted(acc_val_ConvTime_KernelHeight_3.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val kernel height = 3')

l = sorted(err_val_ConvTime_KernelHeight_10.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val kernel height = 10')
l = sorted(acc_val_ConvTime_KernelHeight_10.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val kernel height = 10')

l = sorted(err_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_1.plot(x, y, label='err_val baseline (kernel height = 1)')
l = sorted(acc_val_L2_in_all_no_output.items())
x,y = zip(*l)
ax_2.plot(x, y, label='acc_val baseline (kernel height = 1)')

ax_1.legend(loc=0,prop={'size':12})
ax_1.set_xlabel('Epoch number')
ax_2.legend(loc=0,prop={'size':12})
ax_2.set_xlabel('Epoch number')

fig_1.savefig('convtimenopooling2.pdf')
