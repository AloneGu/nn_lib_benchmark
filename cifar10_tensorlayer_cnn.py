import pickle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d, DropoutLayer, DenseLayer, InputLayer, MaxPool2d, FlattenLayer
import numpy
import time

batch_size = 128
num_classes = 10
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train) = pickle.load(open('data/cifar_data.pkl', 'rb'))
print('x_train shape:', x_train.shape)
x_train = x_train.astype('float32')
x_train = x_train / 127.5
x_train = x_train - 1
y_train = y_train.flatten()
y_train = y_train.astype('int64')
test_cnt = int(len(x_train) * 0.2)
x_test = x_train[:test_cnt]
y_test = y_train[:test_cnt]

print(x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)


def model(x, y_, reuse, is_train=False):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x, name='input')
        net = Conv2d(net, 32, (3, 3), (1, 1), act=tf.nn.relu,
                     padding='SAME', W_init=W_init, name='cnn1')
        net = Conv2d(net, 32, (3, 3), (1, 1), act=tf.nn.relu,
                     W_init=W_init, name='cnn2', padding="VALID")
        net = MaxPool2d(net, name='pool1', padding="VALID")
        net = DropoutLayer(net, keep=0.75, is_train=is_train, name='drop1')

        net = Conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu,
                     padding='SAME', W_init=W_init, name='cnn3')
        net = Conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu,
                     W_init=W_init, name='cnn4', padding="VALID")
        net = MaxPool2d(net, name='pool2', padding="VALID")
        net = DropoutLayer(net, keep=0.75, is_train=is_train, name='drop2')

        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, n_units=512, act=tf.nn.relu,
                         W_init=W_init2, b_init=b_init2, name='d1relu')
        net = DenseLayer(net, n_units=10, act=tf.identity,
                         W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                         name='output')  # output: (batch_size, 10)
        y = net.outputs

        loss = tl.cost.cross_entropy(y, y_, name='cost')

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, loss, acc


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
is_train_ = tf.placeholder(tf.bool, name='is_train_')

## using local response normalization
network, loss, acc = model(x, y_, False)

train_params = network.all_params
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-08, decay=0).minimize(loss, var_list=train_params)
sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)
network.print_layers()
print_freq = 10
print("start training")
for epoch in range(epochs):
    start_time = time.time()
    # train
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=False):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a, is_train_: True})
        err, ac = sess.run([loss, acc], feed_dict={x: X_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("train loss: %f" % (train_loss / n_batch), "   train acc: %f" % (train_acc / n_batch))
    # test
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(x_test, y_test, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: X_test_a, y_: y_test_a})
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("test loss: %f" % (test_loss / n_batch), "   test acc: %f" % (test_acc / n_batch))

    # print info
    print("Epoch %d of %d took %fs" % (epoch + 1, epochs, time.time() - start_time))

sess.close()
