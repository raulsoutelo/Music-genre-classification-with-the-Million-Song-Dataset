import os
import time
import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider
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

train_data = MSD10GenreDataProvider('train', batch_size=50)
valid_data = MSD10GenreDataProvider('valid', batch_size=50)

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')

    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs, weights

def not_fully_connected_layer(inputs, segment_count, segment_dim, num_kernels, nonlinearity=tf.nn.relu):
    weigths = tf.Variable(
        tf.truncated_normal(
            [segment_dim, num_kernels], stddev=2. / (num_kernels + segment_dim) ** 0.5), 
        'weights')  
    biases = tf.Variable(tf.zeros([num_kernels]), 'biases')
    inputs_1 = tf.reshape(inputs, [50, segment_count, segment_dim])
    output = tf.einsum('ijk,kl->ijl', inputs_1, weights) + biases
    temp = tf.reshape(output, [50, segment_count * num_kernels])
    outputs = nonlinearity(temp)  
    return outputs, weigths

def conv_layer(inputs, segment_count, segment_dim, time_length, in_channels, num_kernels, nonlinearity=tf.nn.relu):
    weigths = tf.Variable(
        tf.truncated_normal(
            [time_length, segment_dim, in_channels, num_kernels], stddev=2. / (num_kernels + segment_dim) ** 0.5), 
        'weights')  
    biases = tf.Variable(tf.zeros([num_kernels]), 'biases')
    inputs_1 = tf.reshape(inputs, [50, segment_count, segment_dim, in_channels])
    strides = [1,1,1,1]
    padding = "VALID"
    output_no_bias = tf.nn.conv2d(inputs_1, weigths, strides, padding)
    output = tf.nn.bias_add(output_no_bias, biases)
    temp = tf.reshape(output, [50, (segment_count-(time_length -1)) * num_kernels])
    outputs = nonlinearity(temp)
    return outputs, weigths

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 200
kernels_1=50
kernels_2=50
time_length=5
in_channels=1

with tf.name_scope('layer-1'):
    hidden_1, hidden1_weights = conv_layer(inputs, 120, 25, time_length, 1, kernels_1)
with tf.name_scope('layer-2'):
    hidden_2, hidden2_weights = conv_layer(hidden_1, 120 - (time_length - 1), kernels_1, time_length, 1, kernels_2)
with tf.name_scope('layer-3'):
    hidden_3, hidden3_weights = conv_layer(hidden_2, 120 - 2*(time_length - 1), kernels_2, time_length, 1, kernels_2)
with tf.name_scope('layer-4'):
    hidden_4, hidden4_weights = fully_connected_layer(hidden_3, (120 - 3*(time_length - 1)) * kernels_2, num_hidden)
with tf.name_scope('output-layer'):
    outputs, hidden5_weights = fully_connected_layer(hidden_4, num_hidden, train_data.num_classes, tf.identity)

with tf.name_scope('error'):
    beta = 0.01
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets)
                           + beta * tf.nn.l2_loss(hidden1_weights) 
                           + beta * tf.nn.l2_loss(hidden2_weights) 
                           + beta * tf.nn.l2_loss(hidden3_weights) 
                           + beta * tf.nn.l2_loss(hidden4_weights) 
                          )

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    err_val = {}
    acc_val = {}
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        run_start_time = time.time()
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        run_time = time.time() - run_start_time
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} time={3:.2f}'
              .format(e + 1, running_error, running_accuracy, run_time))
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
        err_val[e + 1] = valid_error
        acc_val[e + 1] = valid_accuracy            
        print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
            .format(valid_error, valid_accuracy))  
         
save_to_file('err_val_convtime_3layers_kernel5',  err_val)             
save_to_file('acc_val_convtime_3layers_kernel5',  acc_val)
