import pickle
from tflearn.data_utils import shuffle, to_categorical

batch_size = 128
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20
steps = 100

# The data, shuffled and split between train and test sets:
(x_train, y_train) = pickle.load(open('data/cifar_data.pkl', 'rb'))
print('x_train shape:', x_train.shape)
y_train = to_categorical(y_train,num_classes)

x_train = x_train.astype('float32')
x_train /= 255

test_cnt = int(len(x_train)*0.2)
x_test = x_train[:test_cnt]
y_test = y_train[:test_cnt]