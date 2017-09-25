import pickle
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np

batch_size = 128
num_classes = 10
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train) = pickle.load(open('data/cifar_data.pkl', 'rb'))
print('x_train shape:', x_train.shape)
x_train = x_train.astype('float32')
x_train = x_train / 127.5
x_train = x_train - 1
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)

test_cnt = int(len(x_train) * 0.2)
x_test = x_train[:test_cnt]
y_test = y_train[:test_cnt]

print('y_train shape:', y_train.shape)


def model(x, y, is_train=False):
    input_layer = x

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    dropout1 = tf.layers.dropout(inputs=pool1, rate=0.25, training=is_train)

    # Convolutional Layer #1
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.25, training=is_train)

    # Dense Layer
    pool2_flat = tf.reshape(dropout2, [-1, 2304])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense1, units=10, activation=tf.nn.softmax, name='output')
    loss = tf.losses.softmax_cross_entropy(y, logits)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, loss, acc


x_ = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x_')
y_ = tf.placeholder(tf.int64, shape=[None, 10], name='y_')
is_train_ = tf.placeholder(tf.bool, name='is_train_')

logits, loss, acc = model(x_, y_, is_train_)

train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-08, decay=0).minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import time


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


for epoch in range(epochs):
    print("---------------")
    start_time = time.time()
    # train
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_a, y_train_a in minibatches(x_train, y_train, batch_size=batch_size):
        sess.run(train_op, feed_dict={x_: X_train_a, y_: y_train_a, is_train_: True})
        err, ac = sess.run([loss, acc], feed_dict={x_: X_train_a, y_: y_train_a, is_train_: False})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("train loss: %f" % (train_loss / n_batch), "   train acc: %f" % (train_acc / n_batch))
    # test
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in minibatches(x_train, y_train, batch_size=batch_size):
        err, ac = sess.run([loss, acc], feed_dict={x_: X_test_a, y_: y_test_a, is_train_: False})
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("test loss: %f" % (test_loss / n_batch), "   test acc: %f" % (test_acc / n_batch))

    # print info
    print("Epoch %d of %d took %fs" % (epoch + 1, epochs, time.time() - start_time))

sess.close()
