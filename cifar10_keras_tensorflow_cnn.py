from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import pickle

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

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

import time

start_t = time.time()
model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])
end_t = time.time()
print("compile done", "time cost: {:.4f}s".format(end_t - start_t))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=2,
          validation_data=(x_test, y_test))
