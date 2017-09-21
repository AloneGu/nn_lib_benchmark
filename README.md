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
14s - loss: 2.2964 - acc: 0.1280 - val_loss: 2.3336 - val_acc: 0.0925
Epoch 2/5
13s - loss: 2.2321 - acc: 0.1850 - val_loss: 2.3313 - val_acc: 0.1750
Epoch 3/5
14s - loss: 2.0164 - acc: 0.2645 - val_loss: 2.0686 - val_acc: 0.2550
Epoch 4/5
14s - loss: 1.9298 - acc: 0.3055 - val_loss: 1.9207 - val_acc: 0.3050
Epoch 5/5
14s - loss: 1.8192 - acc: 0.3310 - val_loss: 1.8015 - val_acc: 0.3600
```

* keras theano

```
Using TensorFlow backend.
Using Theano backend.
x_train shape: (2000, 32, 32, 3)
new x_train shape: (2000, 3, 32, 32)
compile done time cost: 0.0089s
Train on 2000 samples, validate on 400 samples
Epoch 1/5
29s - loss: 2.3033 - acc: 0.1085 - val_loss: 2.4995 - val_acc: 0.1150
Epoch 2/5
29s - loss: 2.2600 - acc: 0.1725 - val_loss: 2.3214 - val_acc: 0.1375
Epoch 3/5
29s - loss: 2.1506 - acc: 0.2375 - val_loss: 1.9750 - val_acc: 0.2950
Epoch 4/5
29s - loss: 1.9895 - acc: 0.2655 - val_loss: 1.7050 - val_acc: 0.3825
Epoch 5/5
29s - loss: 1.9655 - acc: 0.2985 - val_loss: 1.9866 - val_acc: 0.2900
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
Training Step: 16  | total loss: 2.30321 | time: 16.889s
| RMSProp | epoch: 001 | loss: 2.30321 - acc: 0.0848 | val_loss: 2.29459 - val_acc: 0.1050 -- iter: 2000/2000
--
Training Step: 32  | total loss: 2.30299 | time: 17.290s
| RMSProp | epoch: 002 | loss: 2.30299 - acc: 0.1100 | val_loss: 2.29238 - val_acc: 0.1175 -- iter: 2000/2000
--
Training Step: 48  | total loss: 2.29517 | time: 16.596s
| RMSProp | epoch: 003 | loss: 2.29517 - acc: 0.1188 | val_loss: 2.28610 - val_acc: 0.1525 -- iter: 2000/2000
--
Training Step: 64  | total loss: 2.28773 | time: 16.718s
| RMSProp | epoch: 004 | loss: 2.28773 - acc: 0.1201 | val_loss: 2.27897 - val_acc: 0.1175 -- iter: 2000/2000
--
Training Step: 80  | total loss: 2.26888 | time: 16.320s
| RMSProp | epoch: 005 | loss: 2.26888 - acc: 0.1433 | val_loss: 2.25316 - val_acc: 0.1300 -- iter: 2000/2000
--

```

* native tensorflow

* pytorch

```
Using TensorFlow backend.
x_train shape: (2000, 32, 32, 3)
[[6]
 [9]
 [9]
 [4]
 [1]]
[[ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
new x_train shape: (2000, 3, 32, 32)
<class 'torch.autograd.variable.Variable'>
epoch 1
time cost 10.549 train loss 1063.447075843811 train acc 0.7915 test loss 2.302595615386963
------------------------
epoch 2
time cost 10.4082 train loss 36.84132432937622 train acc 0.899 test loss 2.302595615386963
------------------------
epoch 3
time cost 10.3986 train loss 36.84132432937622 train acc 0.899 test loss 2.302595615386963
------------------------
epoch 4
time cost 10.4311 train loss 36.84132432937622 train acc 0.899 test loss 2.302595615386963
------------------------
epoch 5
time cost 10.3178 train loss 36.84132432937622 train acc 0.899 test loss 2.302595615386963
------------------------
```