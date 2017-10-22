import sys
sys.path.insert(0, '/home/raul/Desktop/Million_Song_Dataset')
from datetime import datetime
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

production_100steps = load_from_file('data/production_100steps')

third_layer_features_train = load_from_file('data/third_layer_features_train')
targets_to_store_train = load_from_file('data/targets_to_store_train')
third_layer_features_val = load_from_file('data/third_layer_features_val')
targets_to_store_val = load_from_file('data/targets_to_store_val')

num_hidden = 200
number_classes = 10

inputs = tf.placeholder(tf.float32, [None, num_hidden], 'inputs')
targets = tf.placeholder(tf.float32, [None, number_classes], 'targets')

weights = tf.placeholder(tf.float32, [num_hidden, number_classes])
biases = tf.placeholder(tf.float32, [number_classes])

acc_output = tf.Variable(tf.zeros([number_classes]))

outputs = tf.nn.softmax(tf.matmul(inputs, weights) + biases)

acc_output = tf.add(acc_output, outputs)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('accuracy2'):
    accuracy2 = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(acc_output, 1), tf.argmax(targets, 1)),
            tf.float32))

max_acc = 0

init = tf.global_variables_initializer()

tf.get_default_graph().finalize()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    nwalkers = 2 * (num_hidden * number_classes + number_classes) + 2
    while i < nwalkers:
        t3 = datetime.now()
        weights_and_biases_proposed = production_100steps[i, :]
        weights_proposed =  weights_and_biases_proposed[0:num_hidden*number_classes].reshape((num_hidden, number_classes))
        biases_proposed = weights_and_biases_proposed[num_hidden*number_classes:num_hidden*number_classes + number_classes]

        batch_acc, batch_output, batch_acc_output = sess.run(
            [accuracy, outputs, acc_output],
            feed_dict={inputs: third_layer_features_val, targets: targets_to_store_val,
                       weights: weights_proposed, biases: biases_proposed})

        t5 = datetime.now()
        seconds = (t5 - t3).total_seconds()
        #print batch_acc
        print seconds # only for debugging
        i = i + 1
    total_acc = sess.run(
        [accuracy2],
        feed_dict={inputs: third_layer_features_val, targets: targets_to_store_val,
                   weights: weights_proposed, biases: biases_proposed})

    print 'global accuracy is :' + str(total_acc)
