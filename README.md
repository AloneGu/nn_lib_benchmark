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
Training Step: 16  | total loss: 2.23561 | time: 11.649s
| RMSProp | epoch: 001 | loss: 2.23561 - acc: 0.1553 | val_loss: 2.15268 - val_acc: 0.2525 -- iter: 2000/2000
--
Training Step: 32  | total loss: 2.07924 | time: 11.502s
| RMSProp | epoch: 002 | loss: 2.07924 - acc: 0.2220 | val_loss: 2.05278 - val_acc: 0.2850 -- iter: 2000/2000
--
Training Step: 48  | total loss: 2.08184 | time: 11.294s
| RMSProp | epoch: 003 | loss: 2.08184 - acc: 0.2482 | val_loss: 2.01115 - val_acc: 0.2450 -- iter: 2000/2000
--
Training Step: 64  | total loss: 1.99466 | time: 11.299s
| RMSProp | epoch: 004 | loss: 1.99466 - acc: 0.2754 | val_loss: 1.91739 - val_acc: 0.2975 -- iter: 2000/2000
--
Training Step: 80  | total loss: 1.97854 | time: 11.233s
| RMSProp | epoch: 005 | loss: 1.97854 - acc: 0.2813 | val_loss: 1.89704 - val_acc: 0.3475 -- iter: 2000/2000
--
```

* native tensorflow

* pytorch

```
x_train shape: (2000, 32, 32, 3)
[6 9 9 4 1]
new x_train shape: (2000, 3, 32, 32)
epoch 1
time cost 10.3165 train loss 4.5342 train acc 0.1445
------------------------
epoch 2
time cost 10.2127 train loss 2.1014 train acc 0.226
------------------------
epoch 3
time cost 10.4562 train loss 1.9809 train acc 0.2855
------------------------
epoch 4
time cost 10.2894 train loss 1.8902 train acc 0.3175
------------------------
epoch 5
time cost 10.1774 train loss 1.8126 train acc 0.35
------------------------
```

## results

| lib      | keras theano | keras tensorflow | pytorch | tflearn | tensorflow | tensorlayer |
| ---------|--------------|------------------|---------|---------|------------|-------------|
|epo time(s)|25|9.2|10.29|11.395| | |
|imgs/s|96|260|233|210| | |
|5 epo acc|0.38|0.37|0.35|0.28| | |
|5 epo loss|1.71|1.71|1.81|2.27| | |



              
