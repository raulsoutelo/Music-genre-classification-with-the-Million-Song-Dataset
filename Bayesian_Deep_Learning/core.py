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

weights = tf.Variable(hidden4_bestmodel)
biases = tf.Variable(biases_4_bestmodel)

def fully_connected_layer(inputs, input_dim, output_dim, weights, biases, nonlinearity=tf.nn.relu):
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

num_hidden = 200
number_classes = 10
inputs = tf.placeholder(tf.float32, [None, num_hidden], 'inputs')
targets = tf.placeholder(tf.float32, [None, number_classes], 'targets')

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(inputs, num_hidden, number_classes,
                                                     hidden4_bestmodel, biases_4_bestmodel, tf.nn.softmax)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

# This is the likelihood of the data given the model
with tf.name_scope('likelihood'):
    likelihood = tf.reduce_sum(
            tf.log(tf.reduce_sum(tf.multiply(outputs, targets), 1)))

max_acc = 0
hidden4_bestmodel_max_acc = hidden4_bestmodel
biases_4_bestmodel_max_acc = hidden4_bestmodel

# PRIOR, NOT IMPLEMENTED YET!
#mu = np.zeros(num_hidden*number_classes)
#sigma = np.ones(num_hidden*number_classes)
#gaussian_dist = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)

def lnprob(weights_and_biases_proposed):
    with tf.Session() as sess:
        t3 = datetime.now()
        weights_proposed =  weights_and_biases_proposed[0:num_hidden*number_classes].reshape((num_hidden, number_classes))
        biases_proposed = weights_and_biases_proposed[num_hidden*number_classes:num_hidden*number_classes + number_classes]
        tf.assign(weights, weights_proposed)
        tf.assign(biases, biases_proposed)
        batch_acc, batch_likelihood = sess.run(
            [accuracy, likelihood],
            feed_dict={inputs: third_layer_features_train, targets: targets_to_store_train})
        t5 = datetime.now()
        seconds = (t5 - t3).total_seconds()
        print seconds # only for debugging
        # we could also store the one with the higgest accuracy in the training set!
        #if batch_acc > max_acc:
            #max_acc = batch_acc
            #print 'maximum accuracy achieved in the training set is' + str(batch_acc)
            #hidden4_bestmodel_max_acc = weights_proposed
            #biases_4_bestmodel_max_acc = biases_proposed
        # WE SHOULD INCLUDE HERE THE PRIOR!!
    return batch_likelihood

# We concatenate the weights and the biases
initial_weights_and_biases = np.concatenate((hidden4_bestmodel.reshape(num_hidden*number_classes),biases_4_bestmodel))

nwalkers = 2*(num_hidden*number_classes + number_classes) + 2
p0 = np.zeros((nwalkers, num_hidden*number_classes + number_classes))
#p0 =+ initial_weights_and_biases
for i in range(nwalkers):
    p0[i, :] = initial_weights_and_biases + np.random.normal(0,0.0000001,num_hidden*number_classes + number_classes)

t0 = datetime.now()
sampler = emcee.EnsembleSampler(nwalkers, num_hidden*number_classes + number_classes, lnprob, args=[])
pos, prob, state = sampler.run_mcmc(p0, 1)
t1 = datetime.now()
print 'the time needed to run the preliminary chain (1) is: ' + (t1 - t0).total_seconds()

samples_obtained = sampler.chain
save_to_file('data/samples_one_step_emcee',  samples_obtained)
