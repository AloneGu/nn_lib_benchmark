import pickle
from keras.utils import to_categorical
import tensorflow as tf

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
