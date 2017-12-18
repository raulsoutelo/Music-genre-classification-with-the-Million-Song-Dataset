import sys
sys.path.insert(0, '/home/raul/Desktop/Million_Song_Dataset')
import tensorflow as tf

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

third_layer_features_train = load_from_file('data/third_layer_features_train')
targets_to_store_train = load_from_file('data/targets_to_store_train')
third_layer_features_val = load_from_file('data/third_layer_features_val')
targets_to_store_val = load_from_file('data/targets_to_store_val')

batch_size_variable = 1

def fully_connected_layer(inputs, input_dim, output_dim, weights, biases, nonlinearity=tf.nn.relu):
    weights = tf.constant(weights)
    biases = tf.constant(biases)
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs, weights

num_hidden = 200
number_classes = 10
inputs = tf.placeholder(tf.float32, [None, num_hidden], 'inputs')
targets = tf.placeholder(tf.float32, [None, number_classes], 'targets')

with tf.name_scope('output-layer'):
    outputs, hidden4_weights = fully_connected_layer(inputs, num_hidden, number_classes,
                                                     hidden4_bestmodel, biases_4_bestmodel, tf.identity)

with tf.name_scope('error'):
    beta = 0.01
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    valid_error = 0.
    valid_accuracy = 0.
    batch_error, batch_acc = sess.run(
        [error, accuracy],
        feed_dict={inputs: third_layer_features_val, targets: targets_to_store_val})
    valid_error += batch_error
    valid_accuracy += batch_acc
    print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
        .format(valid_error, valid_accuracy))
