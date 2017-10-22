import sys
sys.path.insert(0, '/home/raul/Desktop/Million_Song_Dataset')
from datetime import datetime
import tensorflow as tf
import numpy as np

import emcee

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

hidden4_bestmodel = load_from_file('data/hidden4_bestmodel')
biases_4_bestmodel= load_from_file('data/biases_4_bestmodel')

third_layer_features_train = load_from_file('data/third_layer_features_train')
targets_to_store_train = load_from_file('data/targets_to_store_train')
third_layer_features_val = load_from_file('data/third_layer_features_val')
targets_to_store_val = load_from_file('data/targets_to_store_val')

batch_size_variable = 1

num_hidden = 200
number_classes = 10

inputs = tf.placeholder(tf.float32, [None, num_hidden], 'inputs')
targets = tf.placeholder(tf.float32, [None, number_classes], 'targets')

weights = tf.placeholder(tf.float32, [num_hidden, number_classes])
biases = tf.placeholder(tf.float32, [number_classes])

outputs = tf.nn.softmax(tf.matmul(inputs, weights) + biases)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

# This is the likelihood of the data given the model
with tf.name_scope('likelihood'):
    likelihood = tf.reduce_sum(
            tf.log(tf.reduce_sum(tf.multiply(outputs, targets), 1)))

# PRIOR, NOT IMPLEMENTED YET!
#mu = np.zeros(num_hidden*number_classes)
#sigma = np.ones(num_hidden*number_classes)
#gaussian_dist = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

tf.get_default_graph().finalize()

def lnprob(weights_and_biases_proposed):
    with tf.Session() as sess:
        t3 = datetime.now()
        weights_proposed =  weights_and_biases_proposed[0:num_hidden*number_classes].reshape((num_hidden, number_classes))
        biases_proposed = weights_and_biases_proposed[num_hidden*number_classes:num_hidden*number_classes + number_classes]
        batch_acc, batch_likelihood = sess.run(
            [accuracy, likelihood],
            feed_dict={inputs: third_layer_features_train, targets: targets_to_store_train,
                       weights: weights_proposed, biases: biases_proposed})
        t5 = datetime.now()
        seconds = (t5 - t3).total_seconds()
        #print seconds # only for debugging
        # WE SHOULD INCLUDE HERE THE PRIOR!!
    return batch_likelihood

print 'definition is fine!'

# We concatenate the weights and the biases
initial_weights_and_biases = np.concatenate((hidden4_bestmodel.reshape(num_hidden*number_classes),biases_4_bestmodel))

nwalkers = 2*(num_hidden*number_classes + number_classes) + 2
p0 = np.zeros((nwalkers, num_hidden*number_classes + number_classes))
for i in range(nwalkers):
    p0[i, :] = initial_weights_and_biases + np.random.normal(0,1,num_hidden*number_classes + number_classes)
sampler = emcee.EnsembleSampler(nwalkers, num_hidden*number_classes + number_classes, lnprob, args=[])

t0 = datetime.now()
pos, prob, state = sampler.run_mcmc(p0, 10)
t1 = datetime.now()
print("Mean acceptance fraction of first chain is: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
print 'the time needed to run the preliminary chain (10) is: ' + str((t1 - t0).total_seconds())
save_to_file('data/preliminary_10steps',  pos)
sampler.reset()

t0 = datetime.now()
pos2, prob2, state2 = sampler.run_mcmc(pos, 100)
t1 = datetime.now()
print("Mean acceptance fraction of second chain is: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
print 'the time needed to run the production chain (100) is: ' + str((t1 - t0).total_seconds())
save_to_file('data/production_100steps',  pos2)
