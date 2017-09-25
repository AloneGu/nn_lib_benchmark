## compare different nn lib

:confused:  :confused:

use 2000 image from cifar 10, test rate = 0.2

run params:

```
batch_size = 128
num_classes = 10
epochs = 5
optimizer = default RMSprop

cpu: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz (8 CPUs), ~2.6GHz
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
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0
=================================================================
Total params: 1,250,858
Trainable params: 1,250,858
Non-trainable params: 0
```

### outputs

* keras theano

```
Using TensorFlow backend.
Using Theano backend.
x_train shape: (2000, 32, 32, 3)
new x_train shape: (2000, 3, 32, 32)
compile done time cost: 0.0269s
Train on 2000 samples, validate on 400 samples
Epoch 1/5
26s - loss: 2.2666 - acc: 0.1645 - val_loss: 2.0803 - val_acc: 0.3325
Epoch 2/5
26s - loss: 1.9582 - acc: 0.2975 - val_loss: 1.8677 - val_acc: 0.3150
Epoch 3/5
25s - loss: 1.8363 - acc: 0.3280 - val_loss: 1.6976 - val_acc: 0.4225
Epoch 4/5
25s - loss: 1.6964 - acc: 0.3715 - val_loss: 1.6402 - val_acc: 0.3950
Epoch 5/5
26s - loss: 1.6284 - acc: 0.4085 - val_loss: 1.5499 - val_acc: 0.4200
```

* keras tensorflow ( my default backend for running gen* scripts )

```
Using TensorFlow backend.
x_train shape: (2000, 32, 32, 3)
compile done time cost: 0.0258s
Train on 2000 samples, validate on 400 samples
9s - loss: 2.2874 - acc: 0.1745 - val_loss: 2.1669 - val_acc: 0.2700
Epoch 2/5
8s - loss: 2.0689 - acc: 0.2360 - val_loss: 2.0315 - val_acc: 0.3025
Epoch 3/5
8s - loss: 1.9458 - acc: 0.2910 - val_loss: 1.9688 - val_acc: 0.2725
Epoch 4/5
9s - loss: 1.8431 - acc: 0.3425 - val_loss: 1.8079 - val_acc: 0.3375
Epoch 5/5
8s - loss: 1.7480 - acc: 0.3740 - val_loss: 1.6333 - val_acc: 0.4075
```

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

* tensorflow

```
Using TensorFlow backend.
x_train shape: (2000, 32, 32, 3)
y_train shape: (2000, 10)
---------------
train loss: 2.252062    train acc: 0.202083
test loss: 2.215203    test acc: 0.246354
Epoch 1 of 5 took 13.886195s
---------------
train loss: 2.176240    train acc: 0.279687
test loss: 2.177576    test acc: 0.281771
Epoch 2 of 5 took 13.705260s
---------------
train loss: 2.134318    train acc: 0.316667
test loss: 2.142393    test acc: 0.315625
Epoch 3 of 5 took 14.268473s
---------------
train loss: 2.128967    train acc: 0.323437
test loss: 2.117443    test acc: 0.340104
Epoch 4 of 5 took 13.845189s
---------------
train loss: 2.102032    train acc: 0.355729
test loss: 2.113987    test acc: 0.339583
Epoch 5 of 5 took 14.666812s

```

* tensorlayer

```
x_train shape: (2000, 32, 32, 3)
(2000, 32, 32, 3) (2000,) float32 int64
  [TL] InputLayer  model/input: (?, 32, 32, 3)
  [TL] Conv2dLayer model/cnn1: shape:[3, 3, 3, 32] strides:[1, 1, 1, 1] pad:SAME act:relu
  [TL] Conv2dLayer model/cnn2: shape:[3, 3, 32, 32] strides:[1, 1, 1, 1] pad:VALID act:relu
  [TL] PoolLayer   model/pool1: ksize:[1, 2, 2, 1] strides:[1, 2, 2, 1] padding:VALID pool:max_pool
  [TL] skip DropoutLayer
  [TL] Conv2dLayer model/cnn3: shape:[3, 3, 32, 64] strides:[1, 1, 1, 1] pad:SAME act:relu
  [TL] Conv2dLayer model/cnn4: shape:[3, 3, 64, 64] strides:[1, 1, 1, 1] pad:VALID act:relu
  [TL] PoolLayer   model/pool2: ksize:[1, 2, 2, 1] strides:[1, 2, 2, 1] padding:VALID pool:max_pool
  [TL] skip DropoutLayer
  [TL] FlattenLayer model/flatten: 2304
  [TL] DenseLayer  model/d1relu: 512 relu
  [TL] DenseLayer  model/output: 10 identity
2017-09-22 17:39:11.667582: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-22 17:39:11.667608: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-22 17:39:11.667632: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
  layer   0: model/cnn1/Relu:0    (?, 32, 32, 32)    float32
  layer   1: model/cnn2/Relu:0    (?, 30, 30, 32)    float32
  layer   2: model/pool1:0        (?, 15, 15, 32)    float32
  layer   3: model/cnn3/Relu:0    (?, 15, 15, 64)    float32
  layer   4: model/cnn4/Relu:0    (?, 13, 13, 64)    float32
  layer   5: model/pool2:0        (?, 6, 6, 64)      float32
  layer   6: model/flatten:0      (?, 2304)          float32
  layer   7: model/d1relu/Relu:0  (?, 512)           float32
  layer   8: model/output/Identity:0 (?, 10)            float32
start training
train loss: 2.226798    train acc: 0.191667
test loss: 2.148892    test acc: 0.223958
Epoch 1 of 5 took 10.944669s
train loss: 2.069909    train acc: 0.239063
test loss: 2.061012    test acc: 0.223958
Epoch 2 of 5 took 10.411697s
train loss: 2.013204    train acc: 0.272396
test loss: 2.060253    test acc: 0.265625
Epoch 3 of 5 took 10.413013s
train loss: 1.966290    train acc: 0.293750
test loss: 1.956488    test acc: 0.291667
Epoch 4 of 5 took 10.517860s
train loss: 1.920005    train acc: 0.304167
test loss: 1.959207    test acc: 0.312500
Epoch 5 of 5 took 10.336267s
```

## results

| lib       | keras theano | keras tensorflow | pytorch | tflearn | tensorflow | tensorlayer |
| ----------|--------------|------------------|---------|---------|------------|-------------|
|epo time(s)|25.6           |10               |10.29    |9.55     |14.1        |10.53|
|imgs/s     |94           |240               |233      |251      |170      |228|
|5 epo acc  |0.41         |0.38              |0.35     |0.26     |0.35        |0.30|
|5 epo loss |1.62         |1.67              |1.81     |2.04     |2.10        |1.92|



              
