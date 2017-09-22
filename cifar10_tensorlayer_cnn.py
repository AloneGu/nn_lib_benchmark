import pickle
from keras.utils import to_categorical
import tensorflow
import tensorlayer as tl
from tensorlayer.layers import *
import numpy

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



def model(x, y_, reuse):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x, name='input')
        net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                    padding='SAME', W_init=W_init, name='cnn1')
        # net = Conv2dLayer(net, act=tf.nn.relu, shape=[5, 5, 3, 64],
        #             strides=[1, 1, 1, 1], padding='SAME',                 # 64 features for each 5x5x3 patch
        #             W_init=W_init, name ='cnn1')           # output: (batch_size, 24, 24, 64)
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool1')
        # net = PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        #             padding='SAME', pool = tf.nn.max_pool, name ='pool1',)# output: (batch_size, 12, 12, 64)
        net = LocalResponseNormLayer(net, depth_radius=4, bias=1.0,
                    alpha=0.001 / 9.0, beta=0.75, name='norm1')
        # net.outputs = tf.nn.lrn(net.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
        #            beta=0.75, name='norm1')

        net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                    padding='SAME', W_init=W_init, name='cnn2')
        # net = Conv2dLayer(net, act=tf.nn.relu, shape=[5, 5, 64, 64],
        #             strides=[1, 1, 1, 1], padding='SAME',                 # 64 features for each 5x5 patch
        #             W_init=W_init, name ='cnn2')           # output: (batch_size, 12, 12, 64)
        net = LocalResponseNormLayer(net, depth_radius=4, bias=1.0,
                    alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # net.outputs = tf.nn.lrn(net.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
        #             beta=0.75, name='norm2')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool2')
        # net = PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        #             padding='SAME', pool = tf.nn.max_pool, name ='pool2') # output: (batch_size, 6, 6, 64)
        net = FlattenLayer(net, name='flatten')                             # output: (batch_size, 2304)
        net = DenseLayer(net, n_units=384, act=tf.nn.relu,
                    W_init=W_init2, b_init=b_init2, name='d1relu')           # output: (batch_size, 384)
        net = DenseLayer(net, n_units=192, act=tf.nn.relu,
                    W_init=W_init2, b_init=b_init2, name='d2relu')           # output: (batch_size, 192)
        net = DenseLayer(net, n_units=10, act=tf.identity,
                    W_init=tf.truncated_normal_initializer(stddev=1/192.0),
                    name='output')                                          # output: (batch_size, 10)
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc




x = tensorflow.placeholder(tensorflow.float32, shape=[None, 24, 24, 3], name='x')
y_ = tensorflow.placeholder(tensorflow.int64, shape=[None, ], name='y_')

## using local response normalization
network, cost, _ = model(x, y_, False)
_, cost_test, acc = model(x, y_, True)


train_params = network.all_params
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
sess = tensorflow.InteractiveSession()
tl.layers.initialize_global_variables(sess)

network.print_params(False)
network.print_layers()
print_freq = 10
for epoch in range(epochs):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, epochs, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(
                                    x_test, y_test, batch_size, shuffle=False):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %f" % (test_loss/ n_batch))
        print("   test acc: %f" % (test_acc/ n_batch))