import pickle
import tflearn
from tflearn.data_utils import shuffle, to_categorical
# from keras.utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.optimizers import RMSProp

batch_size = 128
num_classes = 10
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train) = pickle.load(open('data/cifar_data.pkl', 'rb'))
print('x_train shape:', x_train.shape)
y_train = to_categorical(y_train.flatten(), num_classes)
x_train = x_train.astype('float32')
x_train = x_train / 127.5
x_train = x_train - 1
test_cnt = int(len(x_train) * 0.2)
x_test = x_train[:test_cnt]
y_test = y_train[:test_cnt]

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu', padding='same')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.25)
network = conv_2d(network, 64, 3, activation='relu', padding='same')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.25)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 10, activation='softmax')

opt = RMSProp(learning_rate=0.001, epsilon=1e-08, decay=0)
network = regression(network, optimizer=opt,
                     loss='categorical_crossentropy')

# Train using classifier
import time

start_t = time.time()
model = tflearn.DNN(network, tensorboard_verbose=0)
end_t = time.time()
print("compile done", "time cost: {:.4f}s".format(end_t - start_t))

model.fit(x_train, y_train, n_epoch=epochs, shuffle=True,
          validation_set=(x_test, y_test),
          show_metric=True, batch_size=batch_size,
          run_id='cifar10_cnn')
