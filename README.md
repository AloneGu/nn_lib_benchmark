## compare different nn lib

:confused:  :confused:

use 2000 image from cifar 10, test rate = 0.2

run params:

```
batch_size = 128
num_classes = 10
epochs = 5
optimizer = default RMSprop
```

model structure:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 30, 30, 32)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 15, 15, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 13, 13, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1180160
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0
=================================================================
```

### outputs cpu

cpu: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz (8 CPUs), ~2.6GHz

* keras tensorflow

```
Using TensorFlow backend.
x_train shape: (2000, 32, 32, 3)
compile done time cost: 0.0258s
Train on 2000 samples, validate on 400 samples
10s - loss: 2.2613 - acc: 0.1510 - val_loss: 2.2699 - val_acc: 0.1750
Epoch 2/5
9s - loss: 2.0728 - acc: 0.2580 - val_loss: 1.8698 - val_acc: 0.3400
Epoch 3/5
9s - loss: 1.9295 - acc: 0.3205 - val_loss: 1.7201 - val_acc: 0.3825
Epoch 4/5
9s - loss: 1.7916 - acc: 0.3710 - val_loss: 1.6138 - val_acc: 0.4175
Epoch 5/5
9s - loss: 1.7135 - acc: 0.3745 - val_loss: 1.5312 - val_acc: 0.4450
```

* keras theano

```
Using TensorFlow backend.
Using Theano backend.
x_train shape: (2000, 32, 32, 3)
new x_train shape: (2000, 3, 32, 32)
compile done time cost: 0.0269s
Train on 2000 samples, validate on 400 samples
Epoch 1/5
25s - loss: 2.2166 - acc: 0.1885 - val_loss: 2.1459 - val_acc: 0.1900
Epoch 2/5
25s - loss: 2.0188 - acc: 0.2620 - val_loss: 1.9437 - val_acc: 0.3475
Epoch 3/5
25s - loss: 1.8748 - acc: 0.3150 - val_loss: 1.7416 - val_acc: 0.3675
Epoch 4/5
25s - loss: 1.8143 - acc: 0.3480 - val_loss: 1.6150 - val_acc: 0.4550
Epoch 5/5
25s - loss: 1.7062 - acc: 0.3765 - val_loss: 1.6187 - val_acc: 0.4500
```

* tensorlayer

* tflearn

```
x_train shape: (2000, 32, 32, 3)
compile done time cost: 0.6637s
---------------------------------
Run id: cifar10_cnn
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 2000
Validation samples: 400
--
Training Step: 16  | total loss: 2.30938 | time: 11.888s
| RMSProp | epoch: 001 | loss: 2.30938 - acc: 0.0991 | val_loss: 2.29908 - val_acc: 0.0975 -- iter: 2000/2000
--
Training Step: 32  | total loss: 2.30410 | time: 11.936s
| RMSProp | epoch: 002 | loss: 2.30410 - acc: 0.1071 | val_loss: 2.29590 - val_acc: 0.1025 -- iter: 2000/2000
--
Training Step: 48  | total loss: 2.30410 | time: 11.776s
| RMSProp | epoch: 003 | loss: 2.30410 - acc: 0.1001 | val_loss: 2.28952 - val_acc: 0.1275 -- iter: 2000/2000
--
Training Step: 64  | total loss: 2.29518 | time: 11.597s
| RMSProp | epoch: 004 | loss: 2.29518 - acc: 0.1155 | val_loss: 2.27375 - val_acc: 0.1775 -- iter: 2000/2000
--
Training Step: 80  | total loss: 2.27347 | time: 11.664s
| RMSProp | epoch: 005 | loss: 2.27347 - acc: 0.1466 | val_loss: 2.21062 - val_acc: 0.2475 -- iter: 2000/2000
--
```

* native tensorflow

* pytorch

```
x_train shape: (2000, 32, 32, 3)
[[6]
 [9]
 [9]
 [4]
 [1]]
new x_train shape: (2000, 3, 32, 32)
<class 'torch.autograd.variable.Variable'>
epoch 1
time cost 11.9078 train loss 166434.4122 train acc 0.097
test loss 2.3026
------------------------
epoch 2
time cost 10.8489 train loss 36.8413 train acc 0.101
test loss 2.3026
------------------------
epoch 3
time cost 10.6897 train loss 36.8413 train acc 0.101
test loss 2.3026
------------------------
epoch 4
time cost 10.7904 train loss 36.8413 train acc 0.101
test loss 2.3026
------------------------
epoch 5
time cost 10.8455 train loss 36.8413 train acc 0.101
test loss 2.3026
------------------------
```