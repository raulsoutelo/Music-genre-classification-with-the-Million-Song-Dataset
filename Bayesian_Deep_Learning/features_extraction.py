import sys
sys.path.insert(0, '/home/raul/Desktop/Million_Song_Dataset')
import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider

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

hidden1_bestmodel = load_from_file('data/hidden1_bestmodel')
hidden2_bestmodel = load_from_file('data/hidden2_bestmodel')
hidden3_bestmodel = load_from_file('data/hidden3_bestmodel')
hidden4_bestmodel = load_from_file('data/hidden4_bestmodel')
biases_1_bestmodel= load_from_file('data/biases_1_bestmodel')
biases_2_bestmodel= load_from_file('data/biases_2_bestmodel')
biases_3_bestmodel= load_from_file('data/biases_3_bestmodel')
biases_4_bestmodel= load_from_file('data/biases_4_bestmodel')

batch_size_variable = 1

train_data = MSD10GenreDataProvider('train', batch_size = batch_size_variable)
valid_data = MSD10GenreDataProvider('valid', batch_size = batch_size_variable)

def fully_connected_layer(inputs, input_dim, output_dim, weights, biases, nonlinearity=tf.nn.relu):
    weights = tf.constant(weights)
    biases = tf.constant(biases)
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs, weights

def conv_layer_maxpooling(inputs, image_height, image_width, in_channels, out_channels, kernel_height, kernel_width,
                          weights, biases, nonlinearity=tf.nn.relu):
    weights = tf.constant(weights)
    biases = tf.constant(biases)
    inputs_1 = tf.reshape(inputs, [batch_size_variable, image_height, image_width, in_channels])
    strides = [1, 1, 1, 1]
    padding = "VALID"
    output_no_bias = tf.nn.conv2d(inputs_1, weights, strides, padding)
    output_no_pooling = tf.nn.bias_add(output_no_bias, biases)
    # we add pooling to reduce the dimensionality
    pooling_size = 2
    ksize = [1, pooling_size, 1, 1]
    strides2 = [1, pooling_size, 1, 1]
    output = tf.nn.max_pool(output_no_pooling, ksize, strides2, padding)
    temp = tf.reshape(output, [batch_size_variable, int(np.ceil((image_height - (kernel_height - 1)) / pooling_size)) * out_channels])
    outputs = nonlinearity(temp)
    return outputs, weights

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 200
kernels = 50
kernel_height = 3
in_channels = 1

with tf.name_scope('layer-1'):
    hidden_1, hidden1_weights = conv_layer_maxpooling(inputs, 120, 25, 1, kernels, kernel_height, 25, hidden1_bestmodel,
                                                      biases_1_bestmodel)
with tf.name_scope('layer-2'):
    hidden_2, hidden2_weights = conv_layer_maxpooling(hidden_1, int(np.ceil((120 - (kernel_height - 1)) / 2)), kernels,
                                                      1, kernels, kernel_height, kernels, hidden2_bestmodel,
                                                      biases_2_bestmodel)
with tf.name_scope('layer-3'):
    hidden_3, hidden3_weights = fully_connected_layer(hidden_2, (    int(np.ceil((int(np.ceil((120 - (kernel_height - 1)) / 2)) - (kernel_height - 1)) / 2))) * kernels, num_hidden,
                                                      hidden3_bestmodel, biases_3_bestmodel)
with tf.name_scope('output-layer'):
    outputs, hidden4_weights = fully_connected_layer(hidden_3, num_hidden, train_data.num_classes, hidden4_bestmodel,
                                                     biases_4_bestmodel, tf.identity)

with tf.name_scope('error'):
    beta = 0.01
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets)
                           + beta * tf.nn.l2_loss(hidden1_weights) 
                           + beta * tf.nn.l2_loss(hidden2_weights) 
                           + beta * tf.nn.l2_loss(hidden3_weights) 
                          )

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    err_val = 0
    acc_val = 0
    valid_error = 0.
    valid_accuracy = 0.
    third_layer_features_val = []
    targets_to_store_val = []
    for input_batch, target_batch in valid_data:
        batch_error, batch_acc, batch_hidden_3, batch_targets = sess.run(
            [error, accuracy, hidden_3, targets],
            feed_dict={inputs: input_batch, targets: target_batch})
        third_layer_features_val.append(batch_hidden_3[0,:])
        targets_to_store_val.append(batch_targets[0,:])
        valid_error += batch_error
        valid_accuracy += batch_acc
    third_layer_features_valid = np.array(third_layer_features_val)
    targets_to_store_val_valid = np.array(targets_to_store_val)
    valid_error /= valid_data.num_batches
    valid_accuracy /= valid_data.num_batches
    err_val = valid_error
    acc_val = valid_accuracy
    print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
        .format(valid_error, valid_accuracy))

    third_layer_features_train = []
    targets_to_store_train = []
    for input_batch, target_batch in train_data:
        batch_hidden_3, batch_targets = sess.run(
            [hidden_3, targets],
            feed_dict={inputs: input_batch, targets: target_batch})
        third_layer_features_train.append(batch_hidden_3[0,:])
        targets_to_store_train.append(batch_targets[0,:])
    third_layer_features_train = np.array(third_layer_features_train)
    targets_to_store_val_train = np.array(targets_to_store_train)

save_to_file('data/third_layer_features_train',  third_layer_features_train)
save_to_file('data/targets_to_store_train',  targets_to_store_train)
save_to_file('data/third_layer_features_val',  third_layer_features_val)
save_to_file('data/targets_to_store_val',  targets_to_store_val)